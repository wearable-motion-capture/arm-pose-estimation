import numpy as np

from wear_mocap_ape.utility.names import NNS_TARGETS
from wear_mocap_ape.utility import transformations as ts

FUNCTION_LOOKUP = {
    NNS_TARGETS.ORI_CAL_LARM_UARM_HIPS: lambda a, b: est_to_ori_cal_larm_uarm_hips(a, b),
    NNS_TARGETS.ORI_CAL_LARM_UARM: lambda a, b: est_to_ori_cal_larm_uarm(a, b),
    NNS_TARGETS.ORI_POS_CAL_LARM_UARM_HIPS: lambda a, b: est_ori_pos_cal_larm_uarm_hips(a, b)
}


def msg_from_nn_targets_est(est: np.array, body_measure: np.array, y_targets: NNS_TARGETS):
    return FUNCTION_LOOKUP[y_targets](est, body_measure)


def est_ori_pos_cal_larm_uarm_hips(est: np.array, body_measure: np.array):
    # estimate mean of rotations if we got multiple MC predictions
    if est.shape[0] > 1:
        # Calculate the mean of all predictions mean
        p_hips_quat_g = ts.average_quaternions(est[:, 17:])
        p_larm_quat_g = ts.average_quaternions(est[:, 9:13])
        p_uarm_quat_g = ts.average_quaternions(est[:, 13:17])

        # use body measurements for transitions
        p_uarm_orig_g = np.mean(est[:, 6:9], axis=0)
        p_larm_orig_g = np.mean(est[:, 3:6], axis=0)
        p_hand_orig_g = np.mean(est[:, 0:3], axis=0)
    else:
        p_hand_orig_g = est[0, 0:3]
        p_larm_orig_g = est[0, 3:6]
        p_uarm_orig_g = est[0, 6:9]
        p_larm_quat_g = est[0, 9:13]
        p_uarm_quat_g = est[0, 13:17]
        p_hips_quat_g = est[0, 17:]
    # this is the list for the actual joint positions and rotations
    return np.hstack([
        p_larm_quat_g,  # hand rot [0,1,2,3]
        p_hand_orig_g,  # hand orig [4,5,6]
        p_larm_quat_g,  # larm rot [7,8,9,10]
        p_larm_orig_g,  # larm orig [11,12,13]
        p_uarm_quat_g,  # uarm rot [14,15,16,17]
        p_uarm_orig_g,  # uarm orig [18,19,20]
        p_hips_quat_g  # hips rot [21,22,23,24]
    ])


def est_to_ori_cal_larm_uarm_hips(est: np.array, body_measure: np.array):
    larm_vec, uarm_vec, uarm_orig = body_measure[0, :3], body_measure[0, 3:6], body_measure[0, 6:]

    # estimate mean of rotations if we got multiple MC predictions
    if est.shape[0] > 1:
        # Calculate the mean of all predictions mean
        p_hips_quat_g = ts.average_quaternions(est[:, 17:])
        p_larm_quat_g = ts.average_quaternions(est[:, 9:13])
        p_uarm_quat_g = ts.average_quaternions(est[:, 13:17])

        # use body measurements for transitions
        p_uarm_orig_g = ts.quat_rotate_vector(p_hips_quat_g, uarm_orig)
        p_larm_orig_g = ts.quat_rotate_vector(p_uarm_quat_g, uarm_vec) + p_uarm_orig_g
        p_hand_orig_g = ts.quat_rotate_vector(p_larm_quat_g, larm_vec) + p_larm_orig_g
    else:
        p_hand_orig_g = est[0, 0:3]
        p_larm_orig_g = est[0, 3:6]
        p_uarm_orig_g = est[0, 6:9]
        p_larm_quat_g = est[0, 9:13]
        p_uarm_quat_g = est[0, 13:17]
        p_hips_quat_g = est[0, 17:]

    # this is the list for the actual joint positions and rotations
    return np.hstack([
        p_larm_quat_g,  # hand rot [0,1,2,3]
        p_hand_orig_g,  # hand orig [4,5,6]
        p_larm_quat_g,  # larm rot [7,8,9,10]
        p_larm_orig_g,  # larm orig [11,12,13]
        p_uarm_quat_g,  # uarm rot [14,15,16,17]
        p_uarm_orig_g,  # uarm orig [18,19,20]
        p_hips_quat_g  # hips rot [21,22,23,24]
    ])


def est_to_ori_cal_larm_uarm(est: np.array, body_measure: np.array):
    larm_vec, uarm_vec, uarm_orig = body_measure[0, :3], body_measure[0, 3:6], body_measure[0, 6:]

    # estimate mean of rotations if we got multiple MC predictions
    if est.shape[0] > 1:
        # Calculate the mean of all predictions mean
        p_larm_quat_rh = ts.average_quaternions(est[:, 6:10])
        p_uarm_quat_rh = ts.average_quaternions(est[:, 10:])

        # use body measurements for transitions
        p_larm_orig_rh = ts.quat_rotate_vector(p_uarm_quat_rh, uarm_vec) + uarm_orig
        p_hand_orig_rh = ts.quat_rotate_vector(p_larm_quat_rh, larm_vec) + p_larm_orig_rh
    else:
        p_hand_orig_rh = est[0, :3]
        p_larm_orig_rh = est[0, 3:6]
        p_larm_quat_rh = est[0, 6:10]
        p_uarm_quat_rh = est[0, 10:]

    return np.hstack([
        p_larm_quat_rh,  # hand rot [0,1,2,3]
        p_hand_orig_rh,  # hand orig [4,5,6]
        p_larm_quat_rh,  # larm rot [7,8,9,10]
        p_larm_orig_rh,  # larm orig [11,12,13]
        p_uarm_quat_rh,  # uarm rot [14,15,16,17]
        uarm_orig,  # uarm orig [18,19,20]
        np.array([1, 0, 0, 0])  # hips rot [21,22,23,24]
    ])
