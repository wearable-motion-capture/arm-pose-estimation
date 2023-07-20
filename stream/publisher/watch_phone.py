import logging
import socket
import struct
import time
from datetime import datetime
import queue
import numpy as np

import config
from data_types.bone_map import BoneMap
from utility import transformations
from utility import messaging


class WatchPhonePublisher:
    def __init__(self,
                 port: int,
                 smooth: int = 5,
                 left_hand_mode=True,
                 ip: str = config.IP,
                 tag: str = "PUB WATCH PHONE",
                 bonemap: BoneMap = None):

        self.__tag = tag
        self.__port = port
        self.__ip = ip
        self.__left_hand_mode = left_hand_mode

        # average over multiple time steps
        self.__smooth = smooth
        self.__smooth_hist = []

        # simple lookup for values of interest
        self.__slp = messaging.WATCH_PHONE_IMU_LOOKUP

        # use body measurements for transitions
        if bonemap is None:
            self.__larm_vec = np.array([0, 0, BoneMap.DEFAULT_LARM_LEN])
            self.__uarm_vec = np.array([0, 0, BoneMap.DEFAULT_UARM_LEN])
        else:
            # get values from bone map
            self.__larm_vec = np.array([0, 0, bonemap.left_lower_arm_length])
            self.__uarm_vec = np.array([0, 0, bonemap.left_upper_arm_length])

        self.__udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def calibrate_rotations_from_data(self, row):
        sw_rot_fwd = np.array([
            row[self.__slp["sw_forward_w"]],
            row[self.__slp["sw_forward_x"]],
            row[self.__slp["sw_forward_y"]],
            row[self.__slp["sw_forward_z"]]
        ])
        # smartwatch rotation in our global coord system
        sw_rot = np.array([
            row[self.__slp["sw_rotvec_w"]],
            row[self.__slp["sw_rotvec_x"]],
            row[self.__slp["sw_rotvec_y"]],
            row[self.__slp["sw_rotvec_z"]]
        ])

        # relative watch orientation in global
        sw_quat_cal = transformations.hamilton_product(transformations.quat_invert(sw_rot_fwd), sw_rot)

        ph_rot = np.array([
            row[self.__slp["ph_rotvec_w"]],
            row[self.__slp["ph_rotvec_x"]],
            row[self.__slp["ph_rotvec_y"]],
            row[self.__slp["ph_rotvec_z"]]
        ])
        ph_rot_fwd = np.array([
            row[self.__slp["ph_forward_w"]],
            row[self.__slp["ph_forward_x"]],
            row[self.__slp["ph_forward_y"]],
            row[self.__slp["ph_forward_z"]]
        ])

        ph_quat_cal = transformations.hamilton_product(transformations.quat_invert(ph_rot_fwd), ph_rot)
        return sw_quat_cal, ph_quat_cal

    def row_to_arm_pose(self, row):

        sw_quat_cal, ph_quat_cal = self.calibrate_rotations_from_data(row)

        # decide how to transform coord system depending on arm mode
        if self.__left_hand_mode:
            larm_rot_r = transformations.watch_left_to_global(sw_quat_cal)
            uarm_rot_r = transformations.phone_left_to_global(ph_quat_cal)
        else:
            larm_rot_r = transformations.watch_right_to_global(sw_quat_cal)
            uarm_rot_r = transformations.phone_right_to_global(ph_quat_cal)

        # store rotations in history if smoothing is required
        if self.__smooth > 1:
            self.__smooth_hist.append(np.hstack([larm_rot_r, uarm_rot_r]))
            if len(self.__smooth_hist) < self.__smooth:
                return None
            while len(self.__smooth_hist) > self.__smooth:
                del self.__smooth_hist[0]

            all_rots = np.vstack(self.__smooth_hist)
            # Calculate the mean of all predictions mean
            avg_larm_rot_r = transformations.average_quaternions(all_rots[:, :4])
            avg_uarm_rot_r = transformations.average_quaternions(all_rots[:, 4:])
        else:
            avg_larm_rot_r = larm_rot_r
            avg_uarm_rot_r = uarm_rot_r

        # get the transition from upper arm origin to lower arm origin
        # get transitions from lower arm origin to hand
        larm_origin_rua = transformations.quat_rotate_vector(avg_uarm_rot_r, self.__uarm_vec)
        larm_rotated = transformations.quat_rotate_vector(avg_larm_rot_r, self.__larm_vec)

        hand_origin_rua = larm_rotated + larm_origin_rua
        # this is the list for the actual joint positions and rotations
        return list(
            np.hstack([
                avg_larm_rot_r,  # in this case, hand rotation is larm rotation
                hand_origin_rua,
                avg_larm_rot_r,
                larm_origin_rua,
                avg_uarm_rot_r
            ])
        )

    def send_np_msg(self, msg: np.array) -> int:
        # craft UDP message and send
        msg = struct.pack('f' * len(msg), *msg)
        return self.__udp_socket.sendto(msg, (self.__ip, self.__port))

    def stream_loop(self, sensor_q: queue):
        # used to estimate delta time and processing speed in Hz
        start = datetime.now()
        dat = 0
        logging.info("[{}] starting Unity stream loop".format(self.__tag))

        # this loops while the socket is listening and/or receiving data
        while True:
            # get the most recent smartwatch data row from the queue
            row = sensor_q.get()

            while sensor_q.qsize() > 5:
                row = sensor_q.get()

            # process received data
            if row is not None:
                # second-wise updates to determine message frequency
                now = datetime.now()
                if (now - start).seconds >= 5:
                    start = now
                    logging.info("[{}] {} Hz".format(self.__tag, dat / 5))
                    dat = 0

                # get message as numpy array
                np_msg = self.row_to_arm_pose(row)
                # can return None if not enough historical data for smoothing is available
                if np_msg is None:
                    continue

                # send message to Unity
                self.send_np_msg(msg=np_msg)
                dat += 1
                time.sleep(0.01)
