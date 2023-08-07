import numpy as np

from wear_mocap_ape.utility.names import NNS_TARGETS
from wear_mocap_ape.utility import transformations as ts

FUNCTION_LOOKUP = {
    NNS_TARGETS.ORI_CALIB_UARM_LARM: lambda a, b: uarm_larm_6drr_to_origins(a, b),
    NNS_TARGETS.CONST_XYZ: lambda a, b: hand_larm_xyz_to_origins(a, b),
    NNS_TARGETS.ORI_CALIB_UARM_LARM_HIPS: lambda a, b: uarm_larm_hip_6dof_rh_to_origins_g(a, b)
}


def arm_pose_from_nn_targets(preds: np.array, body_measurements: np.array, y_targets: NNS_TARGETS):
    return FUNCTION_LOOKUP[y_targets](preds, body_measurements)


def uarm_larm_hip_6dof_rh_to_origins_g(preds: np.array, body_measure: np.array):
    """
    :param preds: [uarm_6drr, larm_6drr, hips_sin_cos]
    :param body_measure: [uarm_vec, larm_vec, uarm_orig_rh]
    :return: [hand_orig, larm_orig, uarm_orig, larm_quat_g, uarm_quat_g, hips_quat_g]
    """
    # split combined pred rows back into separate arrays
    uarm_6drr, larm_6drr, hips_sin, hips_cos = preds[:, :6], preds[:, 6:12], preds[:, 12], preds[:, 13]
    uarm_vecs, larm_vecs, uarm_orig_rh = body_measure[:, :3], body_measure[:, 3:6], body_measure[:, 6:]

    # transform to quats
    uarm_quat_rh = ts.six_drr_1x6_to_quat(uarm_6drr)
    larm_quat_rh = ts.six_drr_1x6_to_quat(larm_6drr)
    hips_quat_g = ts.hips_sin_cos_to_quat(hips_sin, hips_cos)

    uarm_quat_g = ts.hamilton_product(hips_quat_g, uarm_quat_rh)
    larm_quat_g = ts.hamilton_product(hips_quat_g, larm_quat_rh)

    # estimate origins in respective reference frame
    p_uarm_orig_g = ts.quat_rotate_vector(hips_quat_g, uarm_orig_rh)  # relative to hips
    p_larm_orig_rua = ts.quat_rotate_vector(uarm_quat_g, uarm_vecs)  # relative to uarm origin
    p_hand_orig_rla = ts.quat_rotate_vector(larm_quat_g, larm_vecs)  # relative to larm origin

    # transform to global positions
    p_larm_orig_g = p_uarm_orig_g + p_larm_orig_rua
    p_hand_orig_g = p_hand_orig_rla + p_larm_orig_g
    return np.hstack([
        p_hand_orig_g,
        p_larm_orig_g,
        p_uarm_orig_g,
        larm_quat_g,
        uarm_quat_g,
        hips_quat_g
    ])


def uarm_larm_6drr_to_origins(preds: np.array, body_measure: np.array):
    """
    takes predictions from a model that has 6dof for lower and upper arm
    as well as upper arm radius as its outputs
    :param preds: [uarm_6drr, larm_6drr]
    :param body_measure: [uarm_vec, larm_vec]
    :return: [hand_orig, larm_orig, larm_quat_rh, uarm_quat_rh]
    """

    # split combined pred rows back into separate arrays
    uarm_6drr, larm_6drr = preds[:, :6], preds[:, 6:]

    # get the default arm vectors from the row-by-row body measurements data
    # lengths may vary if data is shuffled and from different participants
    uarm_vecs = body_measure[:, :3]
    larm_vecs = body_measure[:, 3:]

    # transform 6dof rotation representations back into quaternions
    uarm_rot_mat = ts.six_drr_1x6_to_rot_mat_1x9(uarm_6drr)
    p_uarm_quat_rh = ts.rot_mat_1x9_to_quat(uarm_rot_mat)

    larm_rot_mat = ts.six_drr_1x6_to_rot_mat_1x9(larm_6drr)
    p_larm_quat_rh = ts.rot_mat_1x9_to_quat(larm_rot_mat)

    # get the transition from upper arm origin to lower arm origin
    p_larm_orig_rua = ts.quat_rotate_vector(p_uarm_quat_rh, uarm_vecs)

    # get transitions from lower arm origin to hand
    # RE stands for relative to elbow (lower arm origin)
    rotated_lower_arms_re = ts.quat_rotate_vector(p_larm_quat_rh, larm_vecs)
    p_hand_orig_rua = rotated_lower_arms_re + p_larm_orig_rua

    return np.hstack([
        p_hand_orig_rua,
        p_larm_orig_rua,
        p_larm_quat_rh,
        p_uarm_quat_rh
    ])


def hand_larm_xyz_to_origins(preds: np.array, body_measure: np.array):
    """
    :param preds: [hand_orig_rua, larm_orig_rua]
    :param body_measure: [uarm_vec, larm_vec]
    :return: [hand_origins, lower_arm_origins, lower_arm_rot_rh, upper_arm_rot_rh]
    """
    p_hand_origin_rua = preds[:, :3]
    p_larm_origin_rua = preds[:, 3:]

    uarm_vec = body_measure[:, :3]
    larm_vec = body_measure[:, 3:]

    # estimate correct orientations
    p_larm_quat_rh = ts.quat_a_to_b(larm_vec, p_hand_origin_rua - p_larm_origin_rua)
    p_uarm_quat_rh = ts.quat_a_to_b(uarm_vec, p_larm_origin_rua)

    return np.hstack([
        p_hand_origin_rua,
        p_larm_origin_rua,
        p_larm_quat_rh,
        p_uarm_quat_rh
    ])