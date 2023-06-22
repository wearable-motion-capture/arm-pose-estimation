import logging
import socket
import struct
from datetime import datetime
import queue
import numpy as np

import config
from utility import transformations
from utility import messaging
from utility.transformations import sw_quat_to_global

IP = config.IP
PORT = 50003
TAG = "UDP BROADCAST"
SMOOTHING = 5


def dual_orientations_quat_only(sensor_q: queue):
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # used to estimate delta time and processing speed in Hz
    start = datetime.now()
    dat = 0

    # historical data for time series predictions
    slp = messaging.dual_imu_msg_lookup

    # use body measurements for transitions
    larm_vec = np.array([-0.20, 0, 0])  # for nicer visualisations
    uarm_vec = np.array([-0.25, 0, 0])

    smooth_hist = []

    logging.info("[{}] starting Unity stream loop".format(TAG))
    # this loops while the socket is listening and/or receiving data
    while True:

        # get the most recent smartwatch data row from the queue
        row = None
        while not sensor_q.empty():
            row = sensor_q.get()

        # process received data
        if row is not None:
            # second-wise updates to determine message frequency
            now = datetime.now()
            if (now - start).seconds >= 5:
                start = now
                logging.info("[{}] {} Hz".format(TAG, dat / 5))
                dat = 0

            sw_rot_fwd = np.array([
                row[slp["sw_forward_w"]],
                row[slp["sw_forward_x"]],
                row[slp["sw_forward_y"]],
                row[slp["sw_forward_z"]]
            ])
            # smartwatch rotation in our global coord system
            sw_rot = np.array([
                row[slp["sw_rotvec_w"]],
                row[slp["sw_rotvec_x"]],
                row[slp["sw_rotvec_y"]],
                row[slp["sw_rotvec_z"]]
            ])

            # we assume the watch is at 90 deg on the arm
            watch_quat = np.array([0.7071068, 0, 0.7071068, 0])
            sw_rot_g = sw_quat_to_global(sw_rot)
            sw_quat_calib = transformations.hamilton_product(sw_quat_to_global(sw_rot_fwd), watch_quat)
            sw_quat_r = transformations.hamilton_product(transformations.quat_invert(sw_quat_calib), sw_rot_g)
            larm_rot_r = transformations.hamilton_product(sw_quat_r, watch_quat)

            # the original larm vector points left (negative x), rotate it such that it points forward
            skel_arm_align = transformations.quat_a_to_b(larm_vec, np.array([0, 0, 1]))
            larm_rot_r = transformations.hamilton_product(larm_rot_r, skel_arm_align)  # larm

            ph_rot = np.array([
                row[slp["ph_rotvec_w"]],
                row[slp["ph_rotvec_x"]],
                row[slp["ph_rotvec_y"]],
                row[slp["ph_rotvec_z"]]
            ])
            ph_rot_fwd = np.array([
                row[slp["ph_forward_w"]],
                row[slp["ph_forward_x"]],
                row[slp["ph_forward_y"]],
                row[slp["ph_forward_z"]]
            ])

            # we assume the phone is placed screen outwards and top upwards in a strap on the upper arm
            # the screen should face to the left at a right angle
            phone_quat = transformations.hamilton_product(
                np.array([0.7071068, 0, 0, 0.7071068]),
                np.array([0, 0, 1, 0])
            )
            ph_rot_g = sw_quat_to_global(ph_rot)
            ph_quat_calib = transformations.hamilton_product(sw_quat_to_global(ph_rot_fwd), phone_quat)
            ph_quat_r = transformations.hamilton_product(transformations.quat_invert(ph_quat_calib), ph_rot_g)
            uarm_rot_r = transformations.hamilton_product(ph_quat_r, phone_quat)

            # the original uarm vector points left (negative x), rotate it such that it points forward
            skel_arm_align = transformations.quat_a_to_b(uarm_vec, np.array([0, 0, 1]))
            uarm_rot_r = transformations.hamilton_product(uarm_rot_r, skel_arm_align)

            # store rotations in history if smoothing is required
            if SMOOTHING > 1:
                smooth_hist.append(np.hstack([larm_rot_r, uarm_rot_r]))
                if len(smooth_hist) < SMOOTHING:
                    continue
                while len(smooth_hist) > SMOOTHING:
                    del smooth_hist[0]

                all_rots = np.vstack(smooth_hist)
                # Calculate the mean of all predictions mean
                avg_larm_rot_r = transformations.average_quaternions(all_rots[:, :4])
                avg_uarm_rot_r = transformations.average_quaternions(all_rots[:, 4:])
            else:
                avg_larm_rot_r = larm_rot_r
                avg_uarm_rot_r = uarm_rot_r

            # get the transition from upper arm origin to lower arm origin
            larm_origin_rua = transformations.quat_rotate_vector(avg_uarm_rot_r, uarm_vec)
            # get transitions from lower arm origin to hand
            larm_rotated = transformations.quat_rotate_vector(avg_larm_rot_r, larm_vec)
            hand_origin_rua = larm_rotated + larm_origin_rua

            # this is the list for the actual joint positions and rotations
            basic_value_list = list(
                np.hstack([
                    hand_origin_rua,
                    avg_larm_rot_r,
                    larm_origin_rua,
                    avg_uarm_rot_r
                ])
            )

            # craft UDP message and send
            msg = struct.pack(
                'f' * len(basic_value_list),
                *basic_value_list
            )

            udp_socket.sendto(msg, (IP, PORT))
            dat += 1
