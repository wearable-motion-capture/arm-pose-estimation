import logging
from abc import abstractmethod
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
import queue
import numpy as np

from einops import rearrange

from wear_mocap_ape.estimate import kalman_models
from wear_mocap_ape import config
from wear_mocap_ape.estimate import estimate_joints
from wear_mocap_ape.data_types.bone_map import BoneMap
from wear_mocap_ape.utility import transformations as ts, data_stats
from wear_mocap_ape.data_types import messaging
from wear_mocap_ape.utility.names import NNS_TARGETS, NNS_INPUTS


class WatchPhonePocketKalman:
    def __init__(self,
                 smooth: int = 1,
                 num_ensemble: int = 32,
                 model_name="SW-model-sept-4",
                 window_size: int = 10,
                 stream_mc: bool = True,
                 tag: str = "KALMAN POCKET PHONE"):

        self.__y_targets = NNS_TARGETS.ORI_CAL_LARM_UARM_HIPS
        self.__x_inputs = NNS_INPUTS.WATCH_HIP
        self.__prev_time = datetime.now()

        self.__tag = tag
        self.__active = False
        self.__smooth = smooth
        self.__smooth_hist = []
        self.__stream_mc = stream_mc
        self.__slp = messaging.WATCH_PHONE_IMU_LOOKUP

        self.__row_hist = []

        self.__uarm_vec = np.array([-BoneMap.DEFAULT_UARM_LEN, 0, 0])
        self.__larm_vec = np.array([-BoneMap.DEFAULT_LARM_LEN, 0, 0])
        self.__shou_orig = BoneMap.DEFAULT_L_SHOU_ORIG_RH
        self.last_msg = None
        self.__normalize = True

        self.__batch_size = 1
        self.__dim_x = 14
        self.__dim_z = 14
        self.__input_size_1 = 22
        self.__num_ensemble = num_ensemble
        self.__win_size = window_size

        self.__model = kalman_models.KalmanSmartwatchModel(
            self.__num_ensemble,
            self.__win_size,
            self.__dim_x,
            self.__dim_z,
            self.__input_size_1
        )

        self.__utils = kalman_models.Utils(self.__num_ensemble, self.__dim_x, self.__dim_z)

        # Check model type
        if not isinstance(self.__model, torch.nn.Module):
            raise TypeError("model must be an instance of nn.Module")
        self.__device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            self.__model.cuda()

        # load normalized data stats if required
        stats = data_stats.get_norm_stats(
            x_inputs=self.__x_inputs,
            y_targets=self.__y_targets
        )

        # data is normalized and has to be transformed with pre-calculated mean and std
        self.__xx_m, self.__xx_s = stats["xx_m"], stats["xx_s"]
        self.__yy_m, self.__yy_s = stats["yy_m"], stats["yy_s"]

        # Load the pretrained model
        if torch.cuda.is_available():
            checkpoint = torch.load(config.PATHS["deploy"] / "kalman" / model_name)
            self.__model.load_state_dict(checkpoint["model"])
        else:
            checkpoint = torch.load(config.PATHS["deploy"] / "kalman" / model_name,
                                    map_location=torch.device("cpu"))
            self.__model.load_state_dict(checkpoint["model"])
        self.__model.eval()

        # INIT MODEL
        # for quicker access we store a matrix with relevant body measurements for quick multiplication
        self.__body_measurements = np.r_[self.__larm_vec, self.__uarm_vec, self.__shou_orig][np.newaxis, :]
        self.__init_step = 0

    def is_active(self):
        return self.__active

    def terminate(self):
        self.__active = False

    def get_body_measurements(self):
        return self.__body_measurements

    def errors_from_file(self, file_p: Path, process_msg: bool = False):
        logging.info(f"[{self.__tag}] processing from file")
        input_state = np.zeros((self.__batch_size, self.__num_ensemble, self.__win_size, self.__dim_x))
        input_state = torch.tensor(input_state, dtype=torch.float32)
        input_state = input_state.to(self.__device)

        hand_l_err, hand_r_err = [], []
        elbow_l_err, elbow_r_err = [], []
        hip_rot_errors = []

        dat = pd.read_csv(file_p)
        for i, row in dat.iterrows():
            ## PREDICTIONS
            xx = row[self.__x_inputs.value].to_numpy()
            t_pred, input_state = self.add_obs_and_make_prediction(xx, input_state)

            est = estimate_joints.arm_pose_from_nn_targets(
                preds=t_pred,
                body_measurements=self.__body_measurements,
                y_targets=self.__y_targets
            )
            # estimate mean of rotations if we got multiple MC predictions
            if est.shape[0] > 1:
                # Calculate the mean of all predictions mean
                p_larm_quat_rh = ts.average_quaternions(est[:, 9:13])
                p_uarm_quat_rh = ts.average_quaternions(est[:, 13:17])
                p_hips_quat_rh = ts.average_quaternions(est[:, 17:])

                # get the transition from upper arm origin to lower arm origin
                p_uarm_orig_rh = ts.quat_rotate_vector(p_hips_quat_rh, self.__shou_orig)
                p_larm_orig_rh = ts.quat_rotate_vector(p_uarm_quat_rh, self.__uarm_vec) + p_uarm_orig_rh
                p_hand_orig_rh = ts.quat_rotate_vector(p_larm_quat_rh, self.__larm_vec) + p_larm_orig_rh
            else:
                p_hand_orig_rh = est[0, 0:3]
                p_larm_orig_rh = est[0, 3:6]
                p_uarm_orig_rh = est[0, 6:9]
                p_larm_quat_rh = est[0, 9:13]
                p_uarm_quat_rh = est[0, 13:17]
                p_hips_quat_rh = est[0, 17:]

            # publish the estimation for our unity visualization
            if process_msg:
                msg = np.hstack([
                    p_larm_quat_rh,
                    p_hand_orig_rh,
                    p_larm_quat_rh,
                    p_larm_orig_rh,
                    p_uarm_quat_rh,
                    p_uarm_orig_rh,
                    p_hips_quat_rh
                ])
                if est.shape[0] > 1:
                    for e_row in est:
                        msg = np.hstack([msg, e_row[:9]])
                self.process_msg(msg)

            # GROUND TRUTH
            yy = row[self.__y_targets.value].to_numpy()[np.newaxis, :]
            est = estimate_joints.arm_pose_from_nn_targets(
                preds=yy,
                body_measurements=self.__body_measurements,
                y_targets=self.__y_targets
            )
            gt_hand_orig_rh = est[0, 0:3]
            gt_larm_orig_rh = est[0, 3:6]
            gt_larm_quat_g = est[0, 9:13]
            gt_uarm_quat_g = est[0, 13:17]
            gt_hips_quat_g = est[0, 17:]

            hand_l_err.append(np.linalg.norm(gt_hand_orig_rh - p_hand_orig_rh) * 100)
            hre = ts.quat_to_euler(gt_larm_quat_g) - ts.quat_to_euler(p_larm_quat_rh)
            hree = np.sum(np.abs(np.degrees(np.arctan2(np.sin(hre), np.cos(hre)))))
            hand_r_err.append(hree)

            elbow_l_err.append(np.linalg.norm(gt_larm_orig_rh - p_larm_orig_rh) * 100)
            ere = ts.quat_to_euler(gt_uarm_quat_g) - ts.quat_to_euler(p_uarm_quat_rh)
            eree = np.sum(np.abs(np.degrees(np.arctan2(np.sin(ere), np.cos(ere)))))
            elbow_r_err.append(eree)

            ydiff = ts.quat_to_euler(gt_hips_quat_g)[1] - ts.quat_to_euler(p_hips_quat_rh)[1]
            err = np.abs(np.degrees(np.arctan2(np.sin(ydiff), np.cos(ydiff))))
            hip_rot_errors.append(err)

            if (int(i) + 1) % 100 == 0:
                logging.info(f"[{self.__tag}] processed {i} rows")

        return hand_l_err, hand_r_err, elbow_l_err, elbow_r_err, hip_rot_errors

    def parse_row_to_xx(self, row):
        # process the data
        # pressure - calibrated initial pressure = relative pressure
        r_pres = row[self.__slp["sw_pres"]] - row[self.__slp["sw_init_pres"]]
        # calibrate smartwatch rotation
        sw_rot = np.array([
            row[self.__slp["sw_rotvec_w"]],
            row[self.__slp["sw_rotvec_x"]],
            row[self.__slp["sw_rotvec_y"]],
            row[self.__slp["sw_rotvec_z"]]
        ])
        sw_fwd = np.array([
            row[self.__slp["sw_forward_w"]],
            row[self.__slp["sw_forward_x"]],
            row[self.__slp["sw_forward_y"]],
            row[self.__slp["sw_forward_z"]]
        ])

        # estimate north quat to align Z-axis of global and android coord system
        r = ts.android_quat_to_global_no_north(sw_fwd)
        y_rot = ts.reduce_global_quat_to_y_rot(r)
        quat_north = ts.euler_to_quat(np.array([0, -y_rot, 0]))
        # calibrate watch
        sw_cal_g = ts.android_quat_to_global(sw_rot, quat_north)
        sw_6drr_cal = ts.quat_to_6drr_1x6(sw_cal_g)

        ph_fwd = np.array([
            row[self.__slp["ph_forward_w"]], row[self.__slp["ph_forward_x"]],
            row[self.__slp["ph_forward_y"]], row[self.__slp["ph_forward_z"]]
        ])
        ph_rot = np.array([
            row[self.__slp["ph_rotvec_w"]], row[self.__slp["ph_rotvec_x"]],
            row[self.__slp["ph_rotvec_y"]], row[self.__slp["ph_rotvec_z"]]
        ])
        # the device orientations if the calib position with left arm forward is perfect
        hips_dst_g = np.array([1, 0, 0, 0])
        ph_rot_g = ts.android_quat_to_global(ph_rot, quat_north)
        ph_fwd_g = ts.android_quat_to_global(ph_fwd, quat_north)
        ph_off_g = ts.hamilton_product(ts.quat_invert(ph_fwd_g), hips_dst_g)
        ph_cal_g = ts.hamilton_product(ph_rot_g, ph_off_g)

        # hip y rotation from phone
        hips_y_rot = ts.reduce_global_quat_to_y_rot(ph_cal_g)
        hips_yrot_cal_sin = np.sin(hips_y_rot)
        hips_yrot_cal_cos = np.cos(hips_y_rot)

        # assemble the entire input vector of one time step
        return np.hstack([
            row[self.__slp["sw_dt"]],
            row[self.__slp["sw_gyro_x"]], row[self.__slp["sw_gyro_y"]], row[self.__slp["sw_gyro_z"]],
            row[self.__slp["sw_lvel_x"]], row[self.__slp["sw_lvel_y"]], row[self.__slp["sw_lvel_z"]],
            row[self.__slp["sw_lacc_x"]], row[self.__slp["sw_lacc_y"]], row[self.__slp["sw_lacc_z"]],
            row[self.__slp["sw_grav_x"]], row[self.__slp["sw_grav_y"]], row[self.__slp["sw_grav_z"]],
            sw_6drr_cal,
            r_pres,
            hips_yrot_cal_sin,
            hips_yrot_cal_cos
        ])

    def msg_from_pred(self, pred):
        est = estimate_joints.arm_pose_from_nn_targets(
            preds=pred,
            body_measurements=self.__body_measurements,
            y_targets=self.__y_targets
        )

        # store est in history if smoothing is required
        if self.__smooth > 1:
            self.__smooth_hist.append(est)
            while len(self.__smooth_hist) < self.__smooth:
                self.__smooth_hist.append(est)
            while len(self.__smooth_hist) > self.__smooth:
                del self.__smooth_hist[0]
            est = np.vstack(self.__smooth_hist)

        # estimate mean of rotations if we got multiple MC predictions
        if est.shape[0] > 1:
            # Calculate the mean of all predictions mean
            p_hips_quat_g = ts.average_quaternions(est[:, 17:])
            p_larm_quat_g = ts.average_quaternions(est[:, 9:13])
            p_uarm_quat_g = ts.average_quaternions(est[:, 13:17])

            # get the transition from upper arm origin to lower arm origin
            p_uarm_orig_g = ts.quat_rotate_vector(p_hips_quat_g, self.__shou_orig)
            p_larm_orig_g = ts.quat_rotate_vector(p_uarm_quat_g, self.__uarm_vec) + p_uarm_orig_g
            p_hand_orig_g = ts.quat_rotate_vector(p_larm_quat_g, self.__larm_vec) + p_larm_orig_g
        else:
            p_hand_orig_g = est[0, 0:3]
            p_larm_orig_g = est[0, 3:6]
            p_uarm_orig_g = est[0, 6:9]
            p_larm_quat_g = est[0, 9:13]
            p_uarm_quat_g = est[0, 13:17]
            p_hips_quat_g = est[0, 17:]

        # this is the list for the actual joint positions and rotations
        msg = np.hstack([
            p_larm_quat_g,  # hand rot
            p_hand_orig_g,  # hand orig
            p_larm_quat_g,  # larm rot
            p_larm_orig_g,  # larm orig
            p_uarm_quat_g,  # uarm rot
            p_uarm_orig_g,  # uarm orig
            p_hips_quat_g  # hips rot
        ])
        # store as last msg for getter
        self.last_msg = msg.copy()

        if self.__stream_mc:
            msg = list(msg)
            # now we attach the monte carlo predictions for XYZ positions
            if est.shape[0] > 1:
                for e_row in est:
                    msg += list(e_row[:6])
        return msg

    def processing_loop(self, sensor_q: queue = None):
        logging.info(f"[{self.__tag}] wearable streaming loop")
        self.__init_step = 0

        # used to estimate delta time and processing speed in Hz
        start = datetime.now()
        self.__prev_time = start
        dat = 0

        # this loops while the socket is listening and/or receiving data
        self.__active = True

        logging.info("loaded")
        input_state = None
        while self.__active:

            # processing speed output
            now = datetime.now()
            if (now - start).seconds >= 5:
                start = now
                logging.info(f"[{self.__tag}] {dat / 5} Hz")
                dat = 0
            self.__prev_time = now

            # get the most recent smartwatch data row from the queue
            row = sensor_q.get()
            while sensor_q.qsize() > 5:
                row = sensor_q.get()

            # finally get predicted positions etc
            xx = self.parse_row_to_xx(row)
            t_pred, input_state = self.add_obs_and_make_prediction(xx, input_state)
            msg = self.msg_from_pred(t_pred)
            self.process_msg(msg)
            dat += 1

    def make_prediction_from_row_hist(self, xx, input_state=None):

        if input_state is None:
            # init some dummy input -> [batch_size, ensemble, timestep, dim_x]
            input_state = np.zeros((self.__batch_size, self.__num_ensemble, self.__win_size, self.__dim_x))
            input_state = torch.tensor(input_state, dtype=torch.float32)
            input_state = input_state.to(self.__device)

        if self.__normalize:
            xx = (xx - self.__xx_m) / self.__xx_s

        xx_seq = rearrange(xx, "(bs seq) (en feat) -> bs seq en feat", bs=1, en=1)
        xx_seq = torch.tensor(xx_seq, dtype=torch.float32).to(self.__device)

        with torch.no_grad():
            output = self.__model(xx_seq, input_state)

        # not enough history yet. Make sensor model predictions until we have a time-window worth of data
        if self.__init_step <= self.__win_size:
            # there will be no prediction in this time window
            pred_ = output[3]
            pred_ = rearrange(pred_, "bs en dim -> (bs en) dim")
            pred_ = self.__utils.format_state(pred_)
            pred_ = rearrange(
                pred_, "(bs en) (k dim) -> bs en k dim", bs=self.__batch_size, k=1
            )
            input_state = torch.cat(
                (input_state[:, :, 1:, :], pred_), axis=2
            )
            input_state = input_state.to(self.__device)
            self.__init_step += 1
            # if on GPU copy the tensor to host memory first
            if self.__device.type == 'cuda':
                smp = output[3].cpu().detach().numpy()[0]
            else:
                smp = output[3].detach().numpy()[0]
            return smp, input_state  # return only sensor model prediction

        ensemble = output[0]  # -> output ensemble
        ensemble_ = rearrange(ensemble, "bs (en k) dim -> bs en k dim", k=1)
        input_state = torch.cat(
            (input_state[:, :, 1:, :], ensemble_), axis=2
        )
        input_state = input_state.to(self.__device)

        # if on GPU copy the tensor to host memory first
        if self.__device.type == 'cuda':
            ensemble = ensemble.cpu()

        # get the output
        t_preds = ensemble.detach().numpy()[0]

        if self.__normalize:
            t_preds = t_preds * self.__yy_s + self.__yy_m

        return t_preds, input_state

    def add_obs_and_make_prediction(self, x, input_state=None):
        self.__row_hist.append(x)
        # if not enough data is available yet, simply repeat the input as a first estimate
        while len(self.__row_hist) < self.__win_size:
            self.__row_hist.append(x)
        # if the history is too long, delete old values
        while len(self.__row_hist) > self.__win_size:
            del self.__row_hist[0]

        t_preds, input_state = self.make_prediction_from_row_hist(np.vstack(self.__row_hist), input_state)
        return t_preds, input_state

    @abstractmethod
    def process_msg(self, msg: np.array):
        return

    @property
    def uarm_orig(self):
        return self.__shou_orig

    @property
    def uarm_vec(self):
        return self.__uarm_vec

    @property
    def larm_vec(self):
        return self.__larm_vec

    @property
    def sequence_len(self):
        return self.__win_size

    @property
    def device(self):
        return self.__device

    @property
    def batch_size(self):
        return self.__batch_size

    @property
    def num_ensemble(self):
        return self.__num_ensemble

    @property
    def win_size(self):
        return self.__win_size

    @property
    def y_targets(self):
        return self.__y_targets

    @property
    def x_inputs(self):
        return self.__x_inputs

    @property
    def body_measurements(self):
        return self.__body_measurements

    @property
    def dim_x(self):
        return self.__dim_x
