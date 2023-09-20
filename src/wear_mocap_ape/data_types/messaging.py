"""
These IDs will be collected as ground truth. Position and Rotation
"""
MOTIVE_BONE_IDS = {
    "Hips": 1,
    # No need for these atm.
    # We use the hips as the reference point
    # "Spine": 2,
    # "Chest": 3,
    "LeftShoulder": 6,
    "LeftUpperArm": 7,
    "LeftLowerArm": 8,
    "LeftHand": 9
}

"""
Sensor values from the smartwatch come in a fixed order.
This lookup returns the position of specific measurements.
"""
WATCH_ONLY_IMU_LOOKUP = {
    # timestamp
    "sw_dt": 0,  # delta time since last obs
    "sw_h": 1,  # hour
    "sw_m": 2,  # minute
    "sw_s": 3,  # second
    "sw_ns": 4,  # nanosecond

    # orientation
    "sw_rotvec_w": 5,
    "sw_rotvec_x": 6,
    "sw_rotvec_y": 7,
    "sw_rotvec_z": 8,
    "sw_rotvec_conf": 9,

    # gyro
    "sw_gyro_x": 10,
    "sw_gyro_y": 11,
    "sw_gyro_z": 12,

    # velocity from lacc (1x integrated)
    "sw_lvel_x": 13,
    "sw_lvel_y": 14,
    "sw_lvel_z": 15,

    # linear acceleration
    "sw_lacc_x": 16,
    "sw_lacc_y": 17,
    "sw_lacc_z": 18,

    # atmospheric pressure
    "sw_pres": 19,

    # gravity
    "sw_grav_x": 20,
    "sw_grav_y": 21,
    "sw_grav_z": 22,

    # calibration measurements
    # forward facing direction
    "sw_forward_w": 23,
    "sw_forward_x": 24,
    "sw_forward_y": 25,
    "sw_forward_z": 26,

    # pressure at chest height
    "sw_init_pres": 27
}
watch_only_imu_msg_len = len(WATCH_ONLY_IMU_LOOKUP) * 4

WATCH_PHONE_PPG_LOOKUP = {
    # timestamp
    "sw_h": 0,  # hour
    "sw_m": 1,  # minute
    "sw_s": 2,  # second
    "sw_ns": 3,  # nanosecond

    # PPG sensor
    "hr_raw_00": 4,
    "hr_raw_01": 5,
    "hr_raw_02": 6,
    "hr_raw_03": 7,
    "hr_raw_04": 8,
    "hr_raw_05": 9,
    "hr_raw_06": 10,
    "hr_raw_07": 11,
    "hr_raw_08": 12,
    "hr_raw_09": 13,
    "hr_raw_10": 14,
    "hr_raw_11": 15,
    "hr_raw_12": 16,
    "hr_raw_13": 17,
    "hr_raw_14": 18,
    "hr_raw_15": 19
}

WATCH_PHONE_IMU_LOOKUP = {
    # smartwatch data
    "sw_dt": 0,  # delta time since last obs
    "sw_h": 1,  # hour
    "sw_m": 2,  # minute
    "sw_s": 3,  # second
    "sw_ns": 4,  # nanosecond

    # orientation
    "sw_rotvec_w": 5,
    "sw_rotvec_x": 6,
    "sw_rotvec_y": 7,
    "sw_rotvec_z": 8,
    "sw_rotvec_conf": 9,

    # gyro
    "sw_gyro_x": 10,
    "sw_gyro_y": 11,
    "sw_gyro_z": 12,

    # velocity from lacc (1x integrated)
    "sw_lvel_x": 13,
    "sw_lvel_y": 14,
    "sw_lvel_z": 15,

    # linear acceleration
    "sw_lacc_x": 16,
    "sw_lacc_y": 17,
    "sw_lacc_z": 18,

    # atmospheric pressure
    "sw_pres": 19,

    # gravity
    "sw_grav_x": 20,
    "sw_grav_y": 21,
    "sw_grav_z": 22,

    # phone data
    "ph_dt": 23,  # delta time since last obs
    "ph_h": 24,  # hour
    "ph_m": 25,  # minute
    "ph_s": 26,  # second
    "ph_ns": 27,  # nanosecond

    # orientation
    "ph_rotvec_w": 28,
    "ph_rotvec_x": 29,
    "ph_rotvec_y": 30,
    "ph_rotvec_z": 31,
    "ph_rotvec_conf": 32,

    # gyro
    "ph_gyro_x": 33,
    "ph_gyro_y": 34,
    "ph_gyro_z": 35,

    # velocity from lacc (1x integrated)
    "ph_lvel_x": 36,
    "ph_lvel_y": 37,
    "ph_lvel_z": 38,

    # linear acceleration
    "ph_lacc_x": 39,
    "ph_lacc_y": 40,
    "ph_lacc_z": 41,

    # atmospheric pressure
    "ph_pres": 42,

    # gravity
    "ph_grav_x": 43,
    "ph_grav_y": 44,
    "ph_grav_z": 45,

    # calibration data
    # smartwatch forward
    "sw_forward_w": 46,
    "sw_forward_x": 47,
    "sw_forward_y": 48,
    "sw_forward_z": 49,

    # phone forward
    "ph_forward_w": 50,
    "ph_forward_x": 51,
    "ph_forward_y": 52,
    "ph_forward_z": 53,

    # calibrated pressure
    "sw_init_pres": 54
}
watch_phone_imu_msg_len = len(WATCH_PHONE_IMU_LOOKUP) * 4  # floats
