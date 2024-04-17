import numpy as np

from wear_mocap_ape.data_types.bone_map import BoneMap
from wear_mocap_ape.estimate.estimator import Estimator
from wear_mocap_ape.utility import transformations as ts
from wear_mocap_ape.data_types import messaging
from wear_mocap_ape.utility.names import NNS_INPUTS, NNS_TARGETS


class WatchPhoneUarm(Estimator):
    def __init__(self,
                 smooth: int = 5,
                 tag: str = "Forward Kinematics",
                 bonemap: BoneMap = None):

        super().__init__(
            x_inputs=NNS_INPUTS.WATCH_PHONE_CAL_ALL,
            y_targets=NNS_TARGETS.ORI_CAL_LARM_UARM,
            smooth=smooth,
            normalize=False,
            seq_len=1,
            add_mc_samples=False,
            tag=tag,
            bonemap=bonemap
        )

        self.__tag = tag

        # simple lookup for values of interest
        self.__slp = messaging.WATCH_PHONE_IMU_LOOKUP

    def calibrate_orientation_quats(self,
                                    sw_quat: np.array,
                                    sw_fwd: np.array,
                                    ph_quat: np.array,
                                    ph_fwd: np.array) -> (np.array, np.array):

        # estimate north quat to align Z-axis of global and android coord system
        quat_north = ts.calib_watch_left_to_north_quat(sw_fwd)
        # the arm orientations if the calib position with left arm forward is perfect
        larm_dst_g = np.array([-0.7071068, 0, -0.7071068, 0])
        uarm_dst_g = np.array([0.7071068, 0, 0.7071068, 0])

        # calibrate watch with offset to perfect position
        sw_rot_g = ts.android_quat_to_global(sw_quat, quat_north)
        sw_fwd_g = ts.android_quat_to_global(sw_fwd, quat_north)
        sw_off_g = ts.hamilton_product(ts.quat_invert(sw_fwd_g), larm_dst_g)
        sw_cal_g = ts.hamilton_product(sw_rot_g, sw_off_g)

        # calibrate phone with offset to perfect position
        ph_rot_g = ts.android_quat_to_global(ph_quat, quat_north)
        ph_fwd_g = ts.android_quat_to_global(ph_fwd, quat_north)
        ph_off_g = ts.hamilton_product(ts.quat_invert(ph_fwd_g), uarm_dst_g)
        ph_cal_g = ts.hamilton_product(ph_rot_g, ph_off_g)

        return sw_cal_g, ph_cal_g

    def parse_row_to_xx(self, row: np.array):
        if not type(row) is np.array:
            row = np.array(row)

        sw_fwd = np.array([
            row[self.__slp["sw_forward_w"]], row[self.__slp["sw_forward_x"]],
            row[self.__slp["sw_forward_y"]], row[self.__slp["sw_forward_z"]]
        ])
        sw_rot = np.array([
            row[self.__slp["sw_rotvec_w"]], row[self.__slp["sw_rotvec_x"]],
            row[self.__slp["sw_rotvec_y"]], row[self.__slp["sw_rotvec_z"]]
        ])
        ph_fwd = np.array([
            row[self.__slp["ph_forward_w"]], row[self.__slp["ph_forward_x"]],
            row[self.__slp["ph_forward_y"]], row[self.__slp["ph_forward_z"]]
        ])

        ph_rot = np.array([
            row[self.__slp["ph_rotvec_w"]], row[self.__slp["ph_rotvec_x"]],
            row[self.__slp["ph_rotvec_y"]], row[self.__slp["ph_rotvec_z"]]
        ])

        sw_sensor_dat = np.array([
            row[self.__slp["sw_dt"]],
            row[self.__slp["sw_gyro_x"]], row[self.__slp["sw_gyro_y"]], row[self.__slp["sw_gyro_z"]],
            row[self.__slp["sw_lvel_x"]], row[self.__slp["sw_lvel_y"]], row[self.__slp["sw_lvel_z"]],
            row[self.__slp["sw_lacc_x"]], row[self.__slp["sw_lacc_y"]], row[self.__slp["sw_lacc_z"]],
            row[self.__slp["sw_grav_x"]], row[self.__slp["sw_grav_y"]], row[self.__slp["sw_grav_z"]]
        ])
        r_pres = row[self.__slp["sw_pres"]] - row[self.__slp["sw_init_pres"]]

        ph_sensor_dat = np.array([
            row[self.__slp["ph_gyro_x"]], row[self.__slp["ph_gyro_y"]], row[self.__slp["ph_gyro_z"]],
            row[self.__slp["ph_lvel_x"]], row[self.__slp["ph_lvel_y"]], row[self.__slp["ph_lvel_z"]],
            row[self.__slp["ph_lacc_x"]], row[self.__slp["ph_lacc_y"]], row[self.__slp["ph_lacc_z"]],
            row[self.__slp["ph_grav_x"]], row[self.__slp["ph_grav_y"]], row[self.__slp["ph_grav_z"]]
        ])

        sw_cal_g, ph_cal_g = self.calibrate_orientation_quats(sw_quat=sw_rot, sw_fwd=sw_fwd,
                                                              ph_quat=ph_rot, ph_fwd=ph_fwd)

        return np.hstack([
            sw_sensor_dat,
            ts.quat_to_6drr_1x6(sw_cal_g),
            r_pres,
            ph_sensor_dat,
            ts.quat_to_6drr_1x6(ph_cal_g)
        ])

    def make_prediction_from_row_hist(self, row_hist):
        return np.c_[row_hist[:, 13:19], row_hist[:, -6:]]