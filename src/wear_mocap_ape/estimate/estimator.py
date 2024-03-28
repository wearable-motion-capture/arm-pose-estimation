import logging
from abc import abstractmethod
from datetime import datetime

import torch
import queue
import numpy as np

from wear_mocap_ape.data_types.bone_map import BoneMap
from wear_mocap_ape.estimate import estimate_joints, compose_msg
from wear_mocap_ape.utility import data_stats
from wear_mocap_ape.utility.names import NNS_INPUTS, NNS_TARGETS


class Estimator:
    def __init__(self,
                 x_inputs: NNS_INPUTS,
                 y_targets: NNS_TARGETS,
                 normalize: bool = True,
                 smooth: int = 1,
                 seq_len: int = 1,
                 stream_mc: bool = True,
                 bonemap: BoneMap = None,
                 tag: str = "Estimator"):

        self.__tag = tag
        self._active = True
        self._prev_time = datetime.now()

        self._y_targets = y_targets
        self._x_inputs = x_inputs

        # load normalized data stats if required
        self._normalize = normalize
        if normalize:
            stats = data_stats.get_norm_stats(
                x_inputs=self._x_inputs,
                y_targets=self._y_targets
            )
            # data is normalized and has to be transformed with pre-calculated mean and std
            self._xx_m, self._xx_s = stats["xx_m"], stats["xx_s"]
            self._yy_m, self._yy_s = stats["yy_m"], stats["yy_s"]

        # average over multiple time steps
        self._smooth = max(1, smooth)  # smooth should not be smaller 1
        self._smooth_hist = []

        # monte carlo predictions
        self._last_msg = None
        self._stream_mc = stream_mc

        self._row_hist = []
        self._sequence_len = max(1, seq_len)  # seq_len should not be smaller 1

        # use body measurements for transitions
        if bonemap is None:
            self._larm_vec = np.array([-BoneMap.DEFAULT_LARM_LEN, 0, 0])
            self._uarm_vec = np.array([-BoneMap.DEFAULT_UARM_LEN, 0, 0])
            self._uarm_orig = BoneMap.DEFAULT_UARM_ORIG_RH
        else:
            # get values from bone map
            self._larm_vec = np.array([-bonemap.left_lower_arm_length, 0, 0])
            self._uarm_vec = np.array([-bonemap.left_upper_arm_length, 0, 0])
            self._uarm_orig = bonemap.left_upper_arm_origin_rh

        # for quicker access we store a matrix with relevant body measurements for quick multiplication
        self._body_measurements = np.r_[self._larm_vec, self._uarm_vec, self._uarm_orig][np.newaxis, :]

        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def set_norm_stats(self, stats: dict):
        """overwrites the default norm stats loaded during the initialization"""
        # data is normalized and has to be transformed with pre-calculated mean and std
        self._xx_m, self._xx_s = stats["xx_m"], stats["xx_s"]
        self._yy_m, self._yy_s = stats["yy_m"], stats["yy_s"]
        logging.info("Replaced norm stats xx m+/-s and yy m+/-s")

    def get_last_msg(self):
        return self._last_msg

    def is_active(self):
        return self._active

    def terminate(self):
        self._active = False

    def reset(self):
        self._active = True
        self._row_hist = []
        self._smooth_hist = []

    def add_xx_to_row_hist_and_make_prediction(self, xx) -> np.array:
        self._row_hist.append(xx)
        # if not enough data is available yet, simply repeat the input as a first estimate
        while len(self._row_hist) < self._sequence_len:
            self._row_hist.append(xx)
        # if the history is too long, delete old values
        while len(self._row_hist) > self._sequence_len:
            del self._row_hist[0]
        xx_hist = np.vstack(self._row_hist)

        if self._normalize:
            xx_hist = (xx_hist - self._xx_m) / self._xx_s

        pred = self.make_prediction_from_row_hist(xx_hist)

        if self._normalize:
            pred = pred * self._yy_s + self._yy_m

        # store est in history if smoothing is required
        if self._smooth > 1:
            self._smooth_hist.append(pred)
            while len(self._smooth_hist) < self._smooth:
                self._smooth_hist.append(pred)
            while len(self._smooth_hist) > self._smooth:
                del self._smooth_hist[0]
            pred = np.vstack(self._smooth_hist)

        return pred

    def msg_from_pred(self, pred: np.array, stream_mc: bool) -> np.array:
        est = estimate_joints.arm_pose_from_nn_targets(
            preds=pred,
            body_measurements=self._body_measurements,
            y_targets=self._y_targets
        )
        msg = compose_msg.msg_from_nn_targets_est(est, self._body_measurements, self._y_targets)

        self._last_msg = msg.copy()
        if stream_mc:
            msg = list(msg)
            # now we attach the monte carlo predictions for XYZ positions
            if est.shape[0] > 1:
                for e_row in est:
                    msg += list(e_row[:6])
        return msg

    def processing_loop(self, sensor_q: queue = None):
        logging.info(f"[{self.__tag}] wearable streaming loop")

        # used to estimate delta time and processing speed in Hz
        start = datetime.now()
        self._prev_time = start
        dat = 0

        # this loops while the socket is listening and/or receiving data
        self._active = True

        self.reset()
        while self._active:

            # processing speed output
            now = datetime.now()
            if (now - start).seconds >= 5:
                start = now
                logging.info(f"[{self.__tag}] {dat / 5} Hz")
                dat = 0
            self._prev_time = now

            # get the most recent smartwatch data row from the queue
            row = sensor_q.get()
            while sensor_q.qsize() > 5:
                row = sensor_q.get()

            # finally get predicted positions etc
            xx = self.parse_row_to_xx(row)
            pred = self.add_xx_to_row_hist_and_make_prediction(xx)
            msg = self.msg_from_pred(pred, self._stream_mc)
            self.process_msg(msg)
            dat += 1

    @abstractmethod
    def make_prediction_from_row_hist(self, xx_hist: np.array) -> np.array:
        return

    @abstractmethod
    def parse_row_to_xx(self, row) -> np.array:
        return

    @abstractmethod
    def process_msg(self, msg: np.array):
        return

    @property
    def sequence_len(self):
        return self._sequence_len

    @property
    def body_measurements(self):
        return self._body_measurements

    @property
    def uarm_orig(self):
        return self._uarm_orig

    @property
    def uarm_vec(self):
        return self._uarm_vec

    @property
    def larm_vec(self):
        return self._larm_vec

    @property
    def device(self):
        return self._device

    @property
    def x_inputs(self):
        return self._x_inputs

    @property
    def y_targets(self):
        return self._y_targets
