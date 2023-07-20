from enum import Enum

class NNS_TARGETS(Enum):
    H_ROTS = [
        "uarm_rot_rh_w",
        "uarm_rot_rh_x",
        "uarm_rot_rh_y",
        "uarm_rot_rh_z",
        "larm_rot_rh_w",
        "larm_rot_rh_x",
        "larm_rot_rh_y",
        "larm_rot_rh_z"
    ]

    H_XYZ = [
        "hand_orig_rua_x",
        "hand_orig_rua_y",
        "hand_orig_rua_z",
        "larm_orig_rua_x",
        "larm_orig_rua_y",
        "larm_orig_rua_z"
    ]


class NNS_INPUTS(Enum):
    H_DEFAULT = [
        "sw_dt",
        "sw_gyro_x",
        "sw_gyro_y",
        "sw_gyro_z",
        "sw_lvel_x",
        "sw_lvel_y",
        "sw_lvel_z",
        "sw_lacc_x",
        "sw_lacc_y",
        "sw_lacc_z",
        "sw_grav_x",
        "sw_grav_y",
        "sw_grav_z",
        "sw_rot_cal_w",
        "sw_rot_cal_x",
        "sw_rot_cal_y",
        "sw_rot_cal_z",
        "pres_cal"
    ]
