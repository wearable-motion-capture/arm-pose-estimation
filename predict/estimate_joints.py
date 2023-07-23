import numpy as np
from utility import transformations


def estimate_hand_larm_origins_from_predictions(preds: np.array, body_measurements: np.array, y_targets: str):
    if y_targets == "H_ROTS":
        estimates = uarm_larm_6dof_to_origins(preds=preds,
                                              body_measurements=body_measurements)
    elif y_targets == "H_XYZ":
        estimates = hand_larm_xyz_to_origins(preds=preds,
                                             body_measurements=body_measurements)
    # elif y_targets == NNS_TARGETS.UARM_LARM_QUAT or y_targets == NNS_TARGETS.H_ROTS:
    #     estimates = larm_uarm_quat_to_origins(preds=preds,
    #                                           body_measurements=body_measurements)
    # elif y_targets == NNS_TARGETS.HAND_LARM_POLAR:
    #     estimates = hand_larm_polar_to_origins(preds=preds,
    #                                            body_measurements=body_measurements)
    else:
        raise UserWarning("estimation for {} not implemented".format(y_targets))
    return estimates

#
# def hand_larm_origins_with_rot(preds: np.array,
#                                lower_arm_rot_rh: np.array,
#                                body_measurements: np.array,
#                                y_targets: list):
#     """
#     Chooses the correct estimation function from the y_targets input. This is a bit slower but convenient for debugging and evaluation.
#     :param preds:
#     :param lower_arm_rot_rh:
#     :param body_measurements: [uarm_vec, larm_vec]
#     :param y_targets:
#     :return:  [hand_origins, lower_arm_origins, lower_arm_rot_rh, upper_arm_rot_rh]
#     """
#
#     if y_targets == NNS_INPUTS.UARM_LARM_6DOF_TARGETS:
#         estimates = uarm_larm_6dof_to_origins(preds=preds,
#                                               body_measurements=body_measurements)
#     elif y_targets == NNS_INPUTS.HAND_LARM_XYZ_TARGETS:
#         estimates = hand_larm_xyz_to_origins(preds=preds,
#                                              body_measurements=body_measurements)
#     elif y_targets == NNS_INPUTS.UARM_LARM_QUAT_TARGETS:
#         estimates = larm_uarm_quat_to_origins(preds=preds,
#                                               body_measurements=body_measurements)
#     elif y_targets == NNS_INPUTS.HAND_LARM_POLAR_TARGETS:
#         estimates = hand_larm_polar_to_origins(preds=preds,
#                                                body_measurements=body_measurements)
#     elif y_targets == NNS_INPUTS.HAND_XYZ_TARGETS:
#         estimates = hand_xyz_to_origins(preds=preds,
#                                         larm_rot_rh=lower_arm_rot_rh,
#                                         body_measurements=body_measurements)
#     elif y_targets == NNS_INPUTS.LARM_XYZ_TARGETS:
#         estimates = larm_xyz_to_origins(preds=preds,
#                                         larm_rot_rh=lower_arm_rot_rh,
#                                         body_measurements=body_measurements)
#     elif y_targets == NNS_INPUTS.HAND_LARM_POLAR_TARGETS:
#         estimates = larm_polar_to_origins(preds=preds,
#                                           larm_rot_rh=lower_arm_rot_rh,
#                                           body_measurements=body_measurements)
#     elif y_targets == NNS_INPUTS.UARM_6DOF_TARGETS:
#         estimates = uarm_6dof_to_origins(preds=preds,
#                                          larm_rot_rh=lower_arm_rot_rh,
#                                          body_measurements=body_measurements)
#     elif y_targets == NNS_INPUTS.UARM_QUAT_TARGETS:
#         estimates = uarm_quat_to_origins(preds=preds,
#                                          larm_rot_rh=lower_arm_rot_rh,
#                                          body_measurements=body_measurements)
#     else:
#         raise UserWarning("estimation for {} not implemented".format(y_targets))
#
#     return estimates


def hand_larm_xyz_to_origins(preds: np.array, body_measurements: np.array):
    """
    :param preds:
    :param body_measurements: [uarm_vec, larm_vec]
    :return: [hand_origins, lower_arm_origins, lower_arm_rot_rh, upper_arm_rot_rh]
    """
    pred_hand_origin_rua = preds[:, :3]
    pred_larm_origin_rua = preds[:, 3:]

    larm_vec = body_measurements[:, 3:]
    uarm_vec = body_measurements[:, :3]

    # estimate correct orientations
    larm_rot_rh = transformations.quat_a_to_b(larm_vec, pred_hand_origin_rua - pred_larm_origin_rua)
    uarm_rot_rh = transformations.quat_a_to_b(uarm_vec, pred_larm_origin_rua)

    return np.hstack([pred_hand_origin_rua, pred_larm_origin_rua, larm_rot_rh, uarm_rot_rh])


def hand_xyz_to_origins(preds: np.array, larm_rot_rh: np.array, body_measurements: np.array):
    """
    :param preds:
    :param body_measurements: [uarm_vec, larm_vec]
    :return: [hand_origins, lower_arm_origins, lower_arm_rot_rh, upper_arm_rot_rh]
    """
    pred_hand_origin_rua = preds
    larm_vecs = body_measurements[:, 3:]
    uarm_vec = body_measurements[:, :3]
    # get transition from hand to elbow by rotating the lower arm
    all_vecs_lower_arm_rot = transformations.quat_rotate_vector(larm_rot_rh, larm_vecs)
    pred_larm_origin_rua = pred_hand_origin_rua - all_vecs_lower_arm_rot
    # estimate correct shoulder orientation
    uarm_rot_rh = transformations.quat_a_to_b(uarm_vec, pred_larm_origin_rua)
    return np.hstack([pred_hand_origin_rua, pred_larm_origin_rua, larm_rot_rh, uarm_rot_rh])


def larm_xyz_to_origins(preds: np.array, larm_rot_rh: np.array, body_measurements: np.array):
    """
    :param preds:
    :param body_measurements: [uarm_vec, larm_vec]
    :return: [hand_origins, lower_arm_origins, lower_arm_rot_rh, upper_arm_rot_rh]
    """
    pred_larm_origin_rua = preds
    larm_vec = body_measurements[:, 3:]
    uarm_vec = body_measurements[:, :3]

    # get transition from elbow to hand by rotating the lower arm
    all_vecs_lower_arm_rot = transformations.quat_rotate_vector(larm_rot_rh, larm_vec)
    pred_hand_origin_rua = pred_larm_origin_rua + all_vecs_lower_arm_rot

    # estimate correct shoulder orientation
    uarm_rot_rh = transformations.quat_a_to_b(uarm_vec, pred_larm_origin_rua)
    return np.hstack([pred_hand_origin_rua, pred_larm_origin_rua, larm_rot_rh, uarm_rot_rh])


def uarm_6dof_to_origins(preds: np.array, larm_rot_rh: np.array, body_measurements: np.array):
    """
    :param preds:
    :param body_measurements: [uarm_vec, larm_vec]
    :return: [hand_origins, lower_arm_origins, lower_arm_rot_rh, upper_arm_rot_rh]
    """

    # get the default arm vectors from the row-by-row body measurements data
    # arm lengths may vary if data is shuffled and from different participants
    uarm_vecs = body_measurements[:, :3]
    larm_vecs = body_measurements[:, 3:]

    # transform 6DOF into quaternions
    rot_mats = transformations.six_drr_1x6_to_rot_mat_1x9(preds)
    uarm_rot_rh = transformations.rot_mat_1x9_to_quat(rot_mats)

    # the default upper arm vector rotated by quaternions derived from 6dof representations
    pred_lower_arm_origin_rua = transformations.quat_rotate_vector(np.array(uarm_rot_rh), uarm_vecs)

    # get transitions from lower arm origin to hand by rotating a vector of lower-arm length
    all_vecs_l_lower_arm_to_hand = transformations.quat_rotate_vector(larm_rot_rh, larm_vecs)
    pred_hand_origin_rua = all_vecs_l_lower_arm_to_hand + pred_lower_arm_origin_rua

    return np.hstack([pred_hand_origin_rua, pred_lower_arm_origin_rua, larm_rot_rh, uarm_rot_rh])


def uarm_larm_6dof_to_origins(preds: np.array, body_measurements: np.array):
    """
    takes predictions from a model that has 6dof for lower and upper arm
    as well as upper arm radius as its outputs
    :param preds:
    :param body_measurements: [uarm_vec, larm_vec]
    :return: [hand_origins, lower_arm_origins, lower_arm_rot_rh, upper_arm_rot_rh]
    """

    # split combined pred rows back into separate arrays
    uarm_6dof, larm_6dof = preds[:, :6], preds[:, 6:]

    # get the default arm vectors from the row-by-row body measurements data
    # lengths may vary if data is shuffled and from different participants
    uarm_vecs = body_measurements[:, :3]
    larm_vecs = body_measurements[:, 3:]

    # transform 6dof rotation representations back into quaternions
    uarm_rot_mat = transformations.six_drr_1x6_to_rot_mat_1x9(uarm_6dof)
    upper_arm_rot_rh = transformations.rot_mat_1x9_to_quat(uarm_rot_mat)

    larm_rot_mat = transformations.six_drr_1x6_to_rot_mat_1x9(larm_6dof)
    lower_arm_rot_rh = transformations.rot_mat_1x9_to_quat(larm_rot_mat)

    # get the transition from upper arm origin to lower arm origin
    pred_lower_arm_origins_rua = transformations.quat_rotate_vector(upper_arm_rot_rh, uarm_vecs)

    # get transitions from lower arm origin to hand
    # RE stands for relative to elbow (lower arm origin)
    rotated_lower_arms_re = transformations.quat_rotate_vector(lower_arm_rot_rh, larm_vecs)
    pred_hand_origins_rua = rotated_lower_arms_re + pred_lower_arm_origins_rua

    return np.hstack([pred_hand_origins_rua, pred_lower_arm_origins_rua, lower_arm_rot_rh, upper_arm_rot_rh])


def larm_polar_to_origins(preds: np.array, larm_rot_rh: np.array, body_measurements: np.array):
    """
    :param larm_rot_rh: the arm rotation from the ext_sw_data
    :param preds: predicted quaternions for uarm
    :param body_measurements: [uarm_vec, larm_vec]
    :return: [hand_origins, lower_arm_origins, lower_arm_rot_rh, upper_arm_rot_rh]
    """
    # split combined pred rows into separate arrays
    angles = preds[:, :2]
    larm_vec = body_measurements[:, 3:]
    uarm_vec = body_measurements[:, :3]

    # add new axis with transpose to make later multiplication possible
    rads = np.linalg.norm(body_measurements[:, :3], axis=1)
    upper_arm_r = np.array(rads)[..., np.newaxis]

    # get transition from shoulder to elbow by rotating a vector of upper-arm length
    # the normalized rotated vector and multiply with arm length
    pred_larm_origin_rua = transformations.polar_angles_to_vec(angles) * upper_arm_r

    # get transitions from lower arm origin to hand by rotating a vector of lower-arm length
    all_vecs_l_lower_arm_to_hand = transformations.quat_rotate_vector(larm_rot_rh, larm_vec)
    pred_hand_origin_rua = all_vecs_l_lower_arm_to_hand + pred_larm_origin_rua

    # estimate correct shoulder orientation
    uarm_rot_rh = transformations.quat_a_to_b(uarm_vec, pred_larm_origin_rua)
    return np.hstack([pred_hand_origin_rua, pred_larm_origin_rua, larm_rot_rh, uarm_rot_rh])


def hand_larm_polar_to_origins(preds: np.array, body_measurements: np.array):
    """
    :param preds: predicted quaternions for uarm
    :param body_measurements: [uarm_vec, larm_vec]
    :return: [hand_origins, lower_arm_origins, lower_arm_rot_rh, upper_arm_rot_rh]
    """
    # split combined pred rows into separate arrays
    h_angles = preds[:, :2]
    l_angles = preds[:, 3:5]

    larm_vec = body_measurements[:, 3:]
    uarm_vec = body_measurements[:, :3]

    # add new axis with transpose to make later multiplication possible
    u_rads = np.linalg.norm(uarm_vec, axis=1)
    uarm_r = np.array(u_rads)[..., np.newaxis]

    l_rads = np.linalg.norm(larm_vec, axis=1)
    larm_r = np.array(l_rads)[..., np.newaxis]

    pred_larm_origin_rua = transformations.polar_angles_to_vec(l_angles) * uarm_r
    pred_hand_origin_rla = transformations.polar_angles_to_vec(h_angles) * larm_r

    larm_rot_rh = transformations.quat_a_to_b(larm_vec, pred_hand_origin_rla)
    uarm_rot_rh = transformations.quat_a_to_b(uarm_vec, pred_larm_origin_rua)

    pred_hand_origin_rua = pred_hand_origin_rla + pred_larm_origin_rua
    return np.hstack([pred_hand_origin_rua, pred_larm_origin_rua, larm_rot_rh, uarm_rot_rh])


def larm_uarm_quat_to_origins(preds: np.array, body_measurements: np.array):
    """
    :param preds: predicted quaternions for uarm and larm
    :param body_measurements: [uarm_vec, larm_vec]
    :return: [hand_origins, lower_arm_origins, lower_arm_rot_rh, upper_arm_rot_rh]
    """

    uarm_quat = preds[:, :4]
    larm_quat = preds[:, 4:]

    # get the default arm vectors from the row-by-row body measurements data
    # lengths may vary if data is shuffled and from different participants
    uarm_vec = body_measurements[:, :3]
    larm_vec = body_measurements[:, 3:]

    # RS stands for relative to shoulder
    pred_larm_origin_rua = transformations.quat_rotate_vector(uarm_quat, uarm_vec)

    # get transitions from lower arm origin to hand by rotating a vector of lower-arm length
    rotated_lower_arms_re = transformations.quat_rotate_vector(larm_quat, larm_vec)
    pred_hand_origins_rua = rotated_lower_arms_re + pred_larm_origin_rua

    return np.hstack([pred_hand_origins_rua, pred_larm_origin_rua, larm_quat, uarm_quat])


def uarm_quat_to_origins(preds: np.array, larm_rot_rh: np.array, body_measurements: np.array):
    """
    :param larm_rot_rh: the arm rotation from the ext_sw_data
    :param preds: predicted quaternions for uarm
    :param body_measurements: [uarm_vec, larm_vec]
    :return: [hand_origins, lower_arm_origins, lower_arm_rot_rh, upper_arm_rot_rh]
    """

    # get the default arm vectors from the row-by-row body measurements data
    # lengths may vary if data is shuffled and from different participants
    uarm_vec = body_measurements[:, :3]
    larm_vec = body_measurements[:, 3:]

    # RS stands for relative to shoulder
    pred_larm_origin_rua = transformations.quat_rotate_vector(preds, uarm_vec)

    # get transitions from lower arm origin to hand by rotating a vector of lower-arm length
    rotated_lower_arms_re = transformations.quat_rotate_vector(larm_rot_rh, larm_vec)
    pred_hand_origins_rua = rotated_lower_arms_re + pred_larm_origin_rua

    # estimate correct shoulder orientation
    uarm_rot_rh = transformations.quat_a_to_b(uarm_vec, pred_larm_origin_rua)
    return np.hstack([pred_hand_origins_rua, pred_larm_origin_rua, larm_rot_rh, uarm_rot_rh])
