from enum import Enum


class NNS_TARGETS(Enum):
    ORI_CALIB_UARM_LARM = [
        "gt_uarm_6drr_rh_1", "gt_uarm_6drr_rh_2", "gt_uarm_6drr_rh_3",  # uarm 6drr
        "gt_uarm_6drr_rh_4", "gt_uarm_6drr_rh_5", "gt_uarm_6drr_rh_6",
        "gt_larm_6drr_rh_1", "gt_larm_6drr_rh_2", "gt_larm_6drr_rh_3",  # larm 6drr
        "gt_larm_6drr_rh_4", "gt_larm_6drr_rh_5", "gt_larm_6drr_rh_6"
    ]
    POS_RH_LARM_HAND = [
        "gt_hand_orig_rua_x", "gt_hand_orig_rua_y", "gt_hand_orig_rua_z",
        "gt_larm_orig_rua_x", "gt_larm_orig_rua_y", "gt_larm_orig_rua_z"
    ]
    POS_RH_LARM_HAND_HIPS = [
        "gt_hand_orig_rua_x", "gt_hand_orig_rua_y", "gt_hand_orig_rua_z",
        "gt_larm_orig_rua_x", "gt_larm_orig_rua_y", "gt_larm_orig_rua_z",
        "gt_hips_yrot_cal_sin", "gt_hips_yrot_cal_cos"  # hips calibrated y rot
    ]
    ORI_RH_UARM_LARM_HIPS = [
        "gt_uarm_6drr_rh_1", "gt_uarm_6drr_rh_2", "gt_uarm_6drr_rh_3",  # uarm 6drr
        "gt_uarm_6drr_rh_4", "gt_uarm_6drr_rh_5", "gt_uarm_6drr_rh_6",
        "gt_larm_6drr_rh_1", "gt_larm_6drr_rh_2", "gt_larm_6drr_rh_3",  # larm 6drr
        "gt_larm_6drr_rh_4", "gt_larm_6drr_rh_5", "gt_larm_6drr_rh_6",
        "gt_hips_yrot_cal_sin", "gt_hips_yrot_cal_cos"  # hips calibrated y rot
    ]
    ORI_CAL_LARM_UARM_HIPS = [
        "gt_larm_6drr_cal_1", "gt_larm_6drr_cal_2", "gt_larm_6drr_cal_3",  # larm 6drr
        "gt_larm_6drr_cal_4", "gt_larm_6drr_cal_5", "gt_larm_6drr_cal_6",
        "gt_uarm_6drr_cal_1", "gt_uarm_6drr_cal_2", "gt_uarm_6drr_cal_3",  # uarm 6drr
        "gt_uarm_6drr_cal_4", "gt_uarm_6drr_cal_5", "gt_uarm_6drr_cal_6",
        "gt_hips_yrot_cal_sin", "gt_hips_yrot_cal_cos"  # hips calibrated y rot
    ]

class NNS_INPUTS(Enum):
    WATCH_ONLY = [
        "sw_dt",
        "sw_gyro_x", "sw_gyro_y", "sw_gyro_z",
        "sw_lvel_x", "sw_lvel_y", "sw_lvel_z",
        "sw_lacc_x", "sw_lacc_y", "sw_lacc_z",
        "sw_grav_x", "sw_grav_y", "sw_grav_z",
        "sw_6drr_cal_1", "sw_6drr_cal_2", "sw_6drr_cal_3",
        "sw_6drr_cal_4", "sw_6drr_cal_5", "sw_6drr_cal_6",
        "sw_pres_cal"
    ]
    WATCH_PH_HIP = [
        "sw_dt",
        "sw_gyro_x", "sw_gyro_y", "sw_gyro_z",
        "sw_lvel_x", "sw_lvel_y", "sw_lvel_z",
        "sw_lacc_x", "sw_lacc_y", "sw_lacc_z",
        "sw_grav_x", "sw_grav_y", "sw_grav_z",
        "sw_6drr_cal_1", "sw_6drr_cal_2", "sw_6drr_cal_3",
        "sw_6drr_cal_4", "sw_6drr_cal_5", "sw_6drr_cal_6",
        "sw_pres_cal",
        "ph_hips_yrot_cal_sin",
        "ph_hips_yrot_cal_cos"

    ]
    WATCH_PHONE_CALIB = [
        "sw_dt",
        "sw_gyro_x", "sw_gyro_y", "sw_gyro_z",
        "sw_lvel_x", "sw_lvel_y", "sw_lvel_z",
        "sw_lacc_x", "sw_lacc_y", "sw_lacc_z",
        "sw_grav_x", "sw_grav_y", "sw_grav_z",
        "sw_6drr_cal_1", "sw_6drr_cal_2", "sw_6drr_cal_3",
        "sw_6drr_cal_4", "sw_6drr_cal_5", "sw_6drr_cal_6",
        "sw_pres_cal",
        "ph_gyro_x", "ph_gyro_y", "ph_gyro_z",
        "ph_lvel_x", "ph_lvel_y", "ph_lvel_z",
        "ph_lacc_x", "ph_lacc_y", "ph_lacc_z",
        "ph_grav_x", "ph_grav_y", "ph_grav_z",
        "ph_6drr_cal_1", "ph_6drr_cal_2", "ph_6drr_cal_3",
        "ph_6drr_cal_4", "ph_6drr_cal_5", "ph_6drr_cal_6"
    ]
