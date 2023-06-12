from enum import Enum


class NNS_TARGETS(Enum):
    EVAL = [
        "MC_LeftHand_pos_rua_x",
        "MC_LeftHand_pos_rua_y",
        "MC_LeftHand_pos_rua_z",
        "MC_LeftLowerArm_pos_rua_x",
        "MC_LeftLowerArm_pos_rua_y",
        "MC_LeftLowerArm_pos_rua_z",
        "MCSW_LeftLowerArm_rot_rh_quat_w",
        "MCSW_LeftLowerArm_rot_rh_quat_x",
        "MCSW_LeftLowerArm_rot_rh_quat_y",
        "MCSW_LeftLowerArm_rot_rh_quat_z",
        "BM_LeftUpperArm_vec_x",
        "BM_LeftUpperArm_vec_y",
        "BM_LeftUpperArm_vec_z",
        "BM_LeftLowerArm_vec_x",
        "BM_LeftLowerArm_vec_y",
        "BM_LeftLowerArm_vec_z",
        "time_s",
        "MC_LeftUpperArm_rot_rh_quat_w",
        "MC_LeftUpperArm_rot_rh_quat_x",
        "MC_LeftUpperArm_rot_rh_quat_y",
        "MC_LeftUpperArm_rot_rh_quat_z",
        "MC_LeftLowerArm_rot_rh_quat_w",
        "MC_LeftLowerArm_rot_rh_quat_x",
        "MC_LeftLowerArm_rot_rh_quat_y",
        "MC_LeftLowerArm_rot_rh_quat_z"
    ]

    PPG = [
        "action",
        "no action"
    ]

    D_HAND_XYZ = [
        "d_MC_LeftHand_pos_rua_x",
        "d_MC_LeftHand_pos_rua_y",
        "d_MC_LeftHand_pos_rua_z"
    ]

    # predict x,y,z of the left hand
    HAND_X = ["MC_LeftHand_pos_rua_x"]
    HAND_Y = ["MC_LeftHand_pos_rua_y"]
    HAND_Z = ["MC_LeftHand_pos_rua_z"]
    HAND_XZ = [
        "MC_LeftHand_pos_rua_x",
        "MC_LeftHand_pos_rua_z"
    ]
    HAND_XYZ = [
        "MC_LeftHand_pos_rua_x",
        "MC_LeftHand_pos_rua_y",
        "MC_LeftHand_pos_rua_z"
    ]
    LARM_XYZ = [
        "MC_LeftLowerArm_pos_rua_x",
        "MC_LeftLowerArm_pos_rua_y",
        "MC_LeftLowerArm_pos_rua_z"
    ]
    # predict x,y,z of the left hand and lower arm origin
    HAND_LARM_XYZ = [
        "MC_LeftHand_pos_rua_x",
        "MC_LeftHand_pos_rua_y",
        "MC_LeftHand_pos_rua_z",
        "MC_LeftLowerArm_pos_rua_x",
        "MC_LeftLowerArm_pos_rua_y",
        "MC_LeftLowerArm_pos_rua_z"
    ]

    # predict polar coord angles of lower arm origin and hand origin
    HAND_LARM_POLAR = [
        "MC_LeftHand_pos_rla_polar_elevation",
        "MC_LeftHand_pos_rla_polar_azimuth",
        "MC_LeftHand_pos_rla_polar_radius",
        "MC_LeftLowerArm_pos_rua_polar_elevation",
        "MC_LeftLowerArm_pos_rua_polar_azimuth",
        "MC_LeftLowerArm_pos_rua_polar_radius"
    ]
    LARM_POLAR = [
        "MC_LeftLowerArm_pos_rua_polar_elevation",
        "MC_LeftLowerArm_pos_rua_polar_azimuth",
        "MC_LeftLowerArm_pos_rua_polar_radius"
    ]
    # predict 6dof rotation representation of upper arm
    UARM_6DOF = [
        "MC_LeftUpperArm_rot_rh_6dof_a11",
        "MC_LeftUpperArm_rot_rh_6dof_a21",
        "MC_LeftUpperArm_rot_rh_6dof_a12",
        "MC_LeftUpperArm_rot_rh_6dof_a22",
        "MC_LeftUpperArm_rot_rh_6dof_a13",
        "MC_LeftUpperArm_rot_rh_6dof_a23"
    ]
    # predict 6dof rotation representation of upper arm and lower arm
    UARM_LARM_6DOF = [
        "MC_LeftUpperArm_rot_rh_6dof_a11",
        "MC_LeftUpperArm_rot_rh_6dof_a21",
        "MC_LeftUpperArm_rot_rh_6dof_a12",
        "MC_LeftUpperArm_rot_rh_6dof_a22",
        "MC_LeftUpperArm_rot_rh_6dof_a13",
        "MC_LeftUpperArm_rot_rh_6dof_a23",
        "MC_LeftLowerArm_rot_rh_6dof_a11",
        "MC_LeftLowerArm_rot_rh_6dof_a21",
        "MC_LeftLowerArm_rot_rh_6dof_a12",
        "MC_LeftLowerArm_rot_rh_6dof_a22",
        "MC_LeftLowerArm_rot_rh_6dof_a13",
        "MC_LeftLowerArm_rot_rh_6dof_a23"
    ]
    # predict quaternion rotation of upper arm
    UARM_QUAT = [
        "MC_LeftUpperArm_rot_rh_quat_w",
        "MC_LeftUpperArm_rot_rh_quat_x",
        "MC_LeftUpperArm_rot_rh_quat_y",
        "MC_LeftUpperArm_rot_rh_quat_z"
    ]
    # predict quaternion rotation of lower and upper arm
    UARM_LARM_QUAT = [
        "MC_LeftUpperArm_rot_rh_quat_w",
        "MC_LeftUpperArm_rot_rh_quat_x",
        "MC_LeftUpperArm_rot_rh_quat_y",
        "MC_LeftUpperArm_rot_rh_quat_z",
        "MC_LeftLowerArm_rot_rh_quat_w",
        "MC_LeftLowerArm_rot_rh_quat_x",
        "MC_LeftLowerArm_rot_rh_quat_y",
        "MC_LeftLowerArm_rot_rh_quat_z"
    ]
    KALMAN_T = [
        "MC_LeftHand_pos_rh_x",
        "MC_LeftHand_pos_rh_y",
        "MC_LeftHand_pos_rh_z",
        "MCSW_LeftLowerArm_rot_rh_6dof_a11",
        "MCSW_LeftLowerArm_rot_rh_6dof_a21",
        "MCSW_LeftLowerArm_rot_rh_6dof_a12",
        "MCSW_LeftLowerArm_rot_rh_6dof_a22",
        "MCSW_LeftLowerArm_rot_rh_6dof_a13",
        "MCSW_LeftLowerArm_rot_rh_6dof_a23"
    ]
    KALMAN_TM1 = [
        "MC_LeftHand_pos_rh_x_tm1",
        "MC_LeftHand_pos_rh_y_tm1",
        "MC_LeftHand_pos_rh_z_tm1",
        "MCSW_LeftLowerArm_rot_rh_6dof_a11_tm1",
        "MCSW_LeftLowerArm_rot_rh_6dof_a21_tm1",
        "MCSW_LeftLowerArm_rot_rh_6dof_a12_tm1",
        "MCSW_LeftLowerArm_rot_rh_6dof_a22_tm1",
        "MCSW_LeftLowerArm_rot_rh_6dof_a13_tm1",
        "MCSW_LeftLowerArm_rot_rh_6dof_a23_tm1"
    ]


class NNS_INPUTS(Enum):
    PPG = [
        "hr_raw_00",
        "hr_raw_01",
        "hr_raw_04",
        "hr_raw_05",
        "hr_raw_06",
        "hr_raw_07",
        "hr_raw_08",
        "sw_rotvec_w",
        "sw_rotvec_x",
        "sw_rotvec_y",
        "sw_rotvec_z"
    ]
    MCSW = [
        "d_time_s",
        "r_pres",
        "MCSW_LeftLowerArm_rot_rh_6dof_a11",
        "MCSW_LeftLowerArm_rot_rh_6dof_a21",
        "MCSW_LeftLowerArm_rot_rh_6dof_a12",
        "MCSW_LeftLowerArm_rot_rh_6dof_a22",
        "MCSW_LeftLowerArm_rot_rh_6dof_a13",
        "MCSW_LeftLowerArm_rot_rh_6dof_a23",
        "lacc_x",
        "lacc_y",
        "lacc_z",
        "gyro_x",
        "gyro_y",
        "gyro_z",
        "grav_x",
        "grav_y",
        "grav_z",
        "BM_LeftLowerArm_length",
        "BM_LeftUpperArm_length"
    ]
    KALMAN = [
        "d_time_s",
        "r_pres",
        "MCSW_LeftLowerArm_rot_rh_6dof_a11",
        "MCSW_LeftLowerArm_rot_rh_6dof_a21",
        "MCSW_LeftLowerArm_rot_rh_6dof_a12",
        "MCSW_LeftLowerArm_rot_rh_6dof_a22",
        "MCSW_LeftLowerArm_rot_rh_6dof_a13",
        "MCSW_LeftLowerArm_rot_rh_6dof_a23",
        "lacc_x_rh",
        "lacc_y_rh",
        "lacc_z_rh",
        'lacc_x',
        'lacc_y',
        'lacc_z',
        "gyro_x",
        "gyro_y",
        "gyro_z",
        "grav_x",
        "grav_y",
        "grav_z",
        "BM_LeftLowerArm_length",
        "BM_LeftUpperArm_length"
    ]
    D_HAND = [
        "d_time_s",
        "r_pres",
        "MCSW_LeftLowerArm_rot_rh_6dof_a11",
        "MCSW_LeftLowerArm_rot_rh_6dof_a21",
        "MCSW_LeftLowerArm_rot_rh_6dof_a12",
        "MCSW_LeftLowerArm_rot_rh_6dof_a22",
        "MCSW_LeftLowerArm_rot_rh_6dof_a13",
        "MCSW_LeftLowerArm_rot_rh_6dof_a23",
        "lacc_x_rh",
        "lacc_y_rh",
        "lacc_z_rh",
        '05s_Lacc_x_rh',
        '05s_Lacc_y_rh',
        '05s_Lacc_z_rh',
        'lacc_x',
        'lacc_y',
        'lacc_z',
        "gyro_x",
        "gyro_y",
        "gyro_z",
        "grav_x",
        "grav_y",
        "grav_z",
        "BM_LeftLowerArm_length",
        "BM_LeftUpperArm_length"
    ]
