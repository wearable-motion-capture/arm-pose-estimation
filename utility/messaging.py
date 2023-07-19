motive_bone_ids = {
    "Hips": 1,
    "Spine": 2,
    "Chest": 3,
    "LeftShoulder": 6,
    "LeftUpperArm": 7,
    "LeftLowerArm": 8,
    "LeftHand": 9
}

"""
Sensor values from the smartwatch come in a fixed order.
This lookup returns the position of specific measurements.
"""
sw_standalone_imu_lookup = {
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

    # gyro
    "sw_gyro_x": 9,
    "sw_gyro_y": 10,
    "sw_gyro_z": 11,

    # velocity from lacc (1x integrated)
    "sw_lvel_x": 12,
    "sw_lvel_y": 13,
    "sw_lvel_z": 14,

    # linear acceleration
    "sw_lacc_x": 15,
    "sw_lacc_y": 16,
    "sw_lacc_z": 17,

    # atmospheric pressure
    "sw_pres": 18,

    # gravity
    "sw_grav_x": 19,
    "sw_grav_y": 20,
    "sw_grav_z": 21,

    # calibration measurements
    "sw_init_pres": 22,  # pressure at chest height
    "sw_north_deg": 23  # forward facing direction
}
sw_standalone_imu_msg_len = len(sw_standalone_imu_lookup) * 4

dual_ppg_msg_lookup = {
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

dual_imu_msg_lookup = {
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

    # gyro
    "sw_gyro_x": 9,
    "sw_gyro_y": 10,
    "sw_gyro_z": 11,

    # velocity from lacc (1x integrated)
    "sw_lvel_x": 12,
    "sw_lvel_y": 13,
    "sw_lvel_z": 14,

    # linear acceleration
    "sw_lacc_x": 15,
    "sw_lacc_y": 16,
    "sw_lacc_z": 17,

    # atmospheric pressure
    "sw_pres": 18,

    # gravity
    "sw_grav_x": 19,
    "sw_grav_y": 20,
    "sw_grav_z": 21,

    # phone data
    "ph_dt": 22,  # delta time since last obs
    "ph_h": 23,  # hour
    "ph_m": 24,  # minute
    "ph_s": 25,  # second
    "ph_ns": 26,  # nanosecond

    # orientation
    "ph_rotvec_w": 27,
    "ph_rotvec_x": 28,
    "ph_rotvec_y": 29,
    "ph_rotvec_z": 30,

    # gyro
    "ph_gyro_x": 31,
    "ph_gyro_y": 32,
    "ph_gyro_z": 33,

    # velocity from lacc (1x integrated)
    "ph_lvel_x": 34,
    "ph_lvel_y": 35,
    "ph_lvel_z": 36,

    # linear acceleration
    "ph_lacc_x": 37,
    "ph_lacc_y": 38,
    "ph_lacc_z": 39,

    # gravity
    "ph_grav_x": 40,
    "ph_grav_y": 41,
    "ph_grav_z": 42,

    # atmospheric pressure
    "ph_pres": 43,

    # calibration data
    # smartwatch forward
    "sw_forward_w": 44,
    "sw_forward_x": 45,
    "sw_forward_y": 46,
    "sw_forward_z": 47,

    # phone forward
    "ph_forward_w": 48,
    "ph_forward_x": 49,
    "ph_forward_y": 50,
    "ph_forward_z": 51,

    # rel pressure
    "sw_rel_pres": 52
}
