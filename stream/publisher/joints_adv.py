import logging
import pickle
import socket
import struct
from datetime import datetime
from pathlib import Path

import torch
import queue
import numpy as np

import config
from predict import estimate_joints, models
from data_types.bone_map import BoneMap
from utility import transformations
from utility import messaging

MONTE_CARLO_SAMPLES = 20
IP = config.IP
PORT = 50003
MC_SMOOTHING = 5
TAG = "JOINTS ADV"


def unity_stream_joints_adv(sensor_q: queue, bonemap: BoneMap, params: dict, stream_monte_carlo=False):
    # load the trained network
    torch.set_default_dtype(torch.float64)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load model from given parameters
    nn_model = models.load_model_from_params(params=params)
    y_targets = params["y_targets"]
    sequence_len = params["sequence_len"]
    normalize = params["normalize"] if "normalize" in params else True

    if normalize:
        f_name = "{}_{}".format(params["x_inputs"].name, y_targets.name)
        f_dir = Path(config.paths["deploy"]) / "data_stats"
        f_path = f_dir / f_name

        if not f_path.exists():
            UserWarning("no stats file available in {}".format(f_path))

        with open(f_path, 'rb') as handle:
            logging.info("loaded data stats from {}".format(f_path))
            dat = pickle.load(handle)
            stats = dat
        # data is normalized and has to be transformed with pre-calculated mean and std
        xx_m, xx_s = stats["xx_m"], stats["xx_s"]
        yy_m, yy_s = stats["yy_m"], stats["yy_s"]

    # for quicker access we store a single row containing the used defaults of the given bone map
    body_measurements = np.repeat(np.array([
        np.hstack([bonemap.left_upper_arm_vec, bonemap.left_lower_arm_vec])
    ]), MONTE_CARLO_SAMPLES * MC_SMOOTHING, axis=0)

    # UDP publisher
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # used to estimate delta time and processing speed in Hz
    start = datetime.now()
    prev_time = datetime.now()
    dat = 0

    # historical data for time series predictions
    row_hist = []  # history of predictions for sequence data
    smooth_hist = []  # smoothing averages over a series of time steps
    slp = messaging.sw_standalone_imu_lookup

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
            delta_t = now - prev_time
            delta_t = delta_t.microseconds * 0.000001
            prev_time = now

            # process the data
            r_pres = row[slp["pres"]] - \
                     row[slp["init_pres"]]  # pressure - calibrated initial pressure = relative pressure
            # smartwatch rotation in our global coord system
            sw_rot = transformations.sw_quat_to_global(
                np.array([
                    row[slp["rotvec_w"]],
                    row[slp["rotvec_x"]],
                    row[slp["rotvec_y"]],
                    row[slp["rotvec_z"]]
                ])
            )

            # quaternion to rotate smartwatch forward towards north
            north_quat = transformations.euler_to_quat(
                np.array([0, np.radians(-row[slp["north_deg"]]), 0], dtype=np.float64)
            )

            # now align to known North Pole position. We have our rh rotation
            sw_rot_quat_rh = transformations.hamilton_product(north_quat, sw_rot)
            sw_larm_rot_quat_rh = transformations.hamilton_product(sw_rot_quat_rh,
                                                                   np.array([0, 0, 1, 0], dtype=np.float64))
            # get 6dof rotation representation
            lower_arm_rot_6dof_rh = transformations.rot_mat_1x9_to_six_drr_1x6(
                transformations.quat_to_rot_mat_1x9(sw_larm_rot_quat_rh)
            )

            # assemble the entire input vector of one time step
            xx = np.hstack([
                delta_t,
                r_pres,
                lower_arm_rot_6dof_rh,
                row[slp["lacc_x"]],
                row[slp["lacc_y"]],
                row[slp["lacc_z"]],
                row[slp["gyro_x"]],
                row[slp["gyro_y"]],
                row[slp["gyro_z"]],
                row[slp["grav_x"]],
                row[slp["grav_y"]],
                row[slp["grav_z"]],
                bonemap.left_lower_arm_length,
                bonemap.left_upper_arm_length
            ])

            if normalize:
                # normalize measurements with pre-calculated mean and std
                xx = (xx - xx_m) / xx_s

            # sequences are used for recurrent nets. Stack time steps along 2nd axis
            row_hist.append(xx)
            if len(row_hist) < sequence_len:
                continue
            while len(row_hist) > sequence_len:
                del row_hist[0]
            xx = np.vstack(row_hist)

            # finally, cast to a torch tensor with batch size 1
            xx = torch.tensor(xx[None, :, :])
            with torch.no_grad():
                # make mote carlo predictions if the model makes use of dropout
                t_preds = nn_model.monte_carlo_predictions(x=xx, n_samples=MONTE_CARLO_SAMPLES)

            # if on GPU copy the tensor to host memory first
            if device.type == 'cuda':
                t_preds = t_preds.cpu()
            t_preds = t_preds.numpy()

            # we are only interested in the last prediction of the sequence
            t_preds = t_preds[:, -1, :]

            if normalize:
                # transform predictions back from normalized labels
                t_preds = t_preds * yy_s + yy_m

            # store t_preds in history if smoothing is required
            if MC_SMOOTHING > 1:
                smooth_hist.append(t_preds)
                if len(smooth_hist) < MC_SMOOTHING:
                    continue
                while len(smooth_hist) > MC_SMOOTHING:
                    del smooth_hist[0]
                t_preds = np.vstack(smooth_hist)

            # finally, estimate hand and lower arm origins from prediction data
            est = estimate_joints.hand_larm_origins(preds=t_preds,
                                                    body_measurements=body_measurements,
                                                    y_targets=y_targets)

            # estimate mean of rotations if we got multiple MC predictions
            if est.shape[0] > 1:
                # Calculate the mean of all predictions mean
                pred_larm_rot_rh = transformations.average_quaternions(est[:, 6:10])
                pred_uarm_rot_rh = transformations.average_quaternions(est[:, 10:])
                # use body measurements for transitions
                uarm_vec = body_measurements[0, :3]
                larm_vec = body_measurements[0, 3:]
                # get the transition from upper arm origin to lower arm origin
                pred_larm_origin_rua = transformations.quat_rotate_vector(pred_uarm_rot_rh, uarm_vec)
                # get transitions from lower arm origin to hand
                rotated_lower_arms_re = transformations.quat_rotate_vector(pred_larm_rot_rh, larm_vec)
                pred_hand_origin_rua = rotated_lower_arms_re + pred_larm_origin_rua
            else:
                pred_hand_origin_rua = est[0, :3]
                pred_larm_origin_rua = est[0, 3:6]
                pred_larm_rot_rh = est[0, 6:10]
                pred_uarm_rot_rh = est[0, 10:]

            # this is the list for the actual joint positions and rotations
            basic_value_list = [
                # prediction
                pred_hand_origin_rua[0],
                pred_hand_origin_rua[1],
                pred_hand_origin_rua[2],

                # from smartwatch data
                pred_larm_rot_rh[0],
                pred_larm_rot_rh[1],
                pred_larm_rot_rh[2],
                pred_larm_rot_rh[3],

                # prediction
                pred_larm_origin_rua[0],
                pred_larm_origin_rua[1],
                pred_larm_origin_rua[2],

                # estimated from larm origin
                pred_uarm_rot_rh[0],
                pred_uarm_rot_rh[1],
                pred_uarm_rot_rh[2],
                pred_uarm_rot_rh[3]
            ]

            # now we attach the monte carlo predictions for XYZ positions
            if stream_monte_carlo:
                if est.shape[0] > 1:
                    for e_row in est:
                        basic_value_list += list(e_row[:6])

            # craft UDP message and send
            msg = struct.pack(
                'f' * len(basic_value_list),
                *basic_value_list
            )

            udp_socket.sendto(msg, (IP, PORT))
            dat += 1
