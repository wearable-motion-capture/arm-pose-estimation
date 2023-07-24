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


class WatchPublisher:
    def __init__(self,
                 model_hash: str,
                 ip: str = config.IP_OWN,
                 port: int = config.PORT_PUB_WATCH_IMU_LEFT,
                 bonemap: BoneMap = None,
                 smooth: int = 5,
                 stream_monte_carlo=True,
                 monte_carlo_samples=25):

        self.__tag = "PUB WATCH"
        self.__port = port
        self.__ip = ip

        # average over multiple time steps
        self.__smooth = smooth

        # monte carlo predictions
        self.__stream_mc = stream_monte_carlo
        self.__mc_samples = monte_carlo_samples

        # use arm length measurements for predictions
        if bonemap is None:
            # default values
            self.__larm_vec = np.array([-BoneMap.DEFAULT_LARM_LEN, 0, 0])
            self.__uarm_vec = np.array([-BoneMap.DEFAULT_UARM_LEN, 0, 0])
        else:
            self.__uarm_l = bonemap.left_upper_arm_length
            self.__uarm_vec = bonemap.left_upper_arm_vec

        self.__udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # load the trained network
        torch.set_default_dtype(torch.float64)
        self.__device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # load model from given parameters
        self.__nn_model, params = models.load_deployed_model_from_hash(hash_str=model_hash)
        self.__y_targets_n = params["y_targets_n"]
        self.__sequence_len = params["sequence_len"]
        self.__normalize = params["normalize"] if "normalize" in params else True

        # load normalized data stats if required
        if self.__normalize:
            f_name = "{}_{}".format(params["x_inputs_n"], self.__y_targets_n)
            f_dir = Path(config.PATHS["deploy"]) / "data_stats"
            f_path = f_dir / f_name

            if not f_path.exists():
                UserWarning("no stats file available in {}".format(f_path))

            with open(f_path, 'rb') as handle:
                logging.info("loaded data stats from {}".format(f_path))
                dat = pickle.load(handle)
                stats = dat
            # data is normalized and has to be transformed with pre-calculated mean and std
            self.__xx_m, self.__xx_s = stats["xx_m"], stats["xx_s"]
            self.__yy_m, self.__yy_s = stats["yy_m"], stats["yy_s"]
        else:
            self.__xx_m, self.__xx_s = 0., 1.
            self.__yy_m, self.__yy_s = 0., 1.

    def stream_loop(self, sensor_q: queue):

        logging.info(f"[{self.__tag}] starting watch publisher")

        # used to estimate delta time and processing speed in Hz
        start = datetime.now()
        prev_time = datetime.now()
        dat = 0

        # historical data for time series predictions
        row_hist = []  # history of predictions for sequence data
        smooth_hist = []  # smoothing averages over a series of time steps

        # simple lookup for values of interest
        slp = messaging.WATCH_ONLY_IMU_LOOKUP

        # for quicker access we store a single row containing the used defaults of the given bone map
        body_measurements = np.repeat(np.array([
            np.hstack([self.__uarm_vec, self.__larm_vec])
        ]), self.__mc_samples * self.__smooth, axis=0)

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
                    logging.info(f"[{self.__tag}] {dat / 5} Hz")
                    dat = 0
                delta_t = now - prev_time
                delta_t = delta_t.microseconds * 0.000001
                prev_time = now

                # process the data
                # pressure - calibrated initial pressure = relative pressure
                r_pres = row[slp["sw_pres"]] - row[slp["sw_init_pres"]]

                # calibrate smartwatch rotation
                sw_rot = np.array([
                    row[slp["sw_rotvec_w"]],
                    row[slp["sw_rotvec_x"]],
                    row[slp["sw_rotvec_y"]],
                    row[slp["sw_rotvec_z"]]
                ])
                sw_fwd = np.array([
                    row[slp["sw_forward_w"]],
                    row[slp["sw_forward_x"]],
                    row[slp["sw_forward_y"]],
                    row[slp["sw_forward_z"]]
                ])
                sw_quat_cal = transformations.hamilton_product(transformations.quat_invert(sw_fwd), sw_rot)

                # get 6dof rotation representation
                sw_6drr_cal = transformations.quat_to_6drr_1x6(sw_quat_cal)

                # assemble the entire input vector of one time step
                xx = np.hstack([
                    delta_t,
                    row[slp["sw_gyro_x"]], row[slp["sw_gyro_y"]], row[slp["sw_gyro_z"]],
                    row[slp["sw_lvel_x"]], row[slp["sw_lvel_y"]], row[slp["sw_lvel_z"]],
                    row[slp["sw_lacc_x"]], row[slp["sw_lacc_y"]], row[slp["sw_lacc_z"]],
                    row[slp["sw_grav_x"]], row[slp["sw_grav_y"]], row[slp["sw_grav_z"]],
                    sw_6drr_cal,
                    r_pres
                ])

                if self.__normalize:
                    # normalize measurements with pre-calculated mean and std
                    xx = (xx - self.__xx_m) / self.__xx_s

                # sequences are used for recurrent nets. Stack time steps along 2nd axis
                row_hist.append(xx)
                if len(row_hist) < self.__sequence_len:
                    continue
                while len(row_hist) > self.__sequence_len:
                    del row_hist[0]
                xx = np.vstack(row_hist)

                # finally, cast to a torch tensor with batch size 1
                xx = torch.tensor(xx[None, :, :])
                with torch.no_grad():
                    # make mote carlo predictions if the model makes use of dropout
                    t_preds = self.__nn_model.monte_carlo_predictions(x=xx, n_samples=self.__mc_samples)

                # if on GPU copy the tensor to host memory first
                if self.__device.type == 'cuda':
                    t_preds = t_preds.cpu()
                t_preds = t_preds.numpy()

                # we are only interested in the last prediction of the sequence
                t_preds = t_preds[:, -1, :]

                if self.__normalize:
                    # transform predictions back from normalized labels
                    t_preds = t_preds * self.__yy_s + self.__yy_m

                # store t_preds in history if smoothing is required
                if self.__smooth > 1:
                    smooth_hist.append(t_preds)
                    if len(smooth_hist) < self.__smooth:
                        continue
                    while len(smooth_hist) > self.__smooth:
                        del smooth_hist[0]
                    t_preds = np.vstack(smooth_hist)

                # finally, estimate hand and lower arm origins from prediction data
                est = estimate_joints.estimate_hand_larm_origins_from_predictions(
                    preds=t_preds,
                    body_measurements=body_measurements,
                    y_targets=self.__y_targets_n
                )

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

                # # hand from watch
                # fwd_to_left = np.array([0.7071068, 0., 0.7071068, 0.])  # a 90deg y rotation
                # hand_rot_r = transformations.watch_left_to_global(sw_quat_cal)
                # hand_rot_r = transformations.hamilton_product(hand_rot_r, fwd_to_left)
                hand_rot_r = pred_larm_rot_rh

                # this is the list for the actual joint positions and rotations
                basic_value_list = [
                    # we pass a hand orientation too, for future work
                    # currently, hand rotation is smartwatch rotation
                    hand_rot_r[0],
                    hand_rot_r[1],
                    hand_rot_r[2],
                    hand_rot_r[3],

                    pred_hand_origin_rua[0],
                    pred_hand_origin_rua[1],
                    pred_hand_origin_rua[2],

                    pred_larm_rot_rh[0],
                    pred_larm_rot_rh[1],
                    pred_larm_rot_rh[2],
                    pred_larm_rot_rh[3],

                    pred_larm_origin_rua[0],
                    pred_larm_origin_rua[1],
                    pred_larm_origin_rua[2],

                    pred_uarm_rot_rh[0],
                    pred_uarm_rot_rh[1],
                    pred_uarm_rot_rh[2],
                    pred_uarm_rot_rh[3]
                ]

                # now we attach the monte carlo predictions for XYZ positions
                if self.__stream_mc:
                    if est.shape[0] > 1:
                        for e_row in est:
                            basic_value_list += list(e_row[:6])

                # craft UDP message and send
                msg = struct.pack(
                    'f' * len(basic_value_list),
                    *basic_value_list
                )

                self.__udp_socket.sendto(msg, (self.__ip, self.__port))
                dat += 1
