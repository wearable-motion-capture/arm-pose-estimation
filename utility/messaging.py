"""
Sensor values from the smartwatch come in a fixed order.
This lookup returns the position of specific measurements.
"""
sw_imu_ppg_msg_lookup = {
    # orientation
    "rotvec_w": 0,
    "rotvec_x": 1,
    "rotvec_y": 2,
    "rotvec_z": 3,

    # linear acceleration
    "lacc_x": 4,
    "lacc_y": 5,
    "lacc_z": 6,

    # atmospheric pressure
    "pres": 7,

    # gravity
    "grav_x": 8,
    "grav_y": 9,
    "grav_z": 10,

    # gyroscope
    "gyro_x": 11,
    "gyro_y": 12,
    "gyro_z": 13,

    # PPG sensor
    "hr_raw_00": 14,
    "hr_raw_01": 15,
    "hr_raw_02": 16,
    "hr_raw_03": 17,
    "hr_raw_04": 18,
    "hr_raw_05": 19,
    "hr_raw_06": 20,
    "hr_raw_07": 21,
    "hr_raw_08": 22,
    "hr_raw_09": 23,
    "hr_raw_10": 24,
    "hr_raw_11": 25,
    "hr_raw_12": 26,
    "hr_raw_13": 27,
    "hr_raw_14": 28,
    "hr_raw_15": 29,

    # calibration measurements
    "init_pres": 30,  # pressure at chest height
    "north_deg": 31  # forward facing direction
}

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
    # timestamp
    "sw_h": 0,  # hour
    "sw_m": 1,  # minute
    "sw_s": 2,  # second
    "sw_ns": 3,  # nanosecond

    # orientation
    "sw_rotvec_w": 4,
    "sw_rotvec_x": 5,
    "sw_rotvec_y": 6,
    "sw_rotvec_z": 7,

    # linear acceleration
    "sw_lacc_x": 8,
    "sw_lacc_y": 9,
    "sw_lacc_z": 10,

    # atmospheric pressure
    "sw_pres": 11,

    # gravity
    "sw_grav_x": 12,
    "sw_grav_y": 13,
    "sw_grav_z": 14,

    # gyroscope
    "sw_gyro_x": 15,
    "sw_gyro_y": 16,
    "sw_gyro_z": 17,

    # phone data
    # timestamp
    "ph_h": 18,  # hour
    "ph_m": 19,  # minute
    "ph_s": 20,  # second
    "ph_ns": 21,  # nanosecond

    # orientation
    "ph_rotvec_w": 22,
    "ph_rotvec_x": 23,
    "ph_rotvec_y": 24,
    "ph_rotvec_z": 25,

    # linear acceleration
    "ph_lacc_x": 26,
    "ph_lacc_y": 27,
    "ph_lacc_z": 28,

    # gravity
    "ph_grav_x": 29,
    "ph_grav_y": 30,
    "ph_grav_z": 31,

    # gyroscope
    "ph_gyro_x": 32,
    "ph_gyro_y": 33,
    "ph_gyro_z": 34,

    # calibration data
    # smartwatch forward
    "sw_forward_w": 35,
    "sw_forward_x": 36,
    "sw_forward_y": 37,
    "sw_forward_z": 38,

    # phone forward
    "ph_forward_w": 39,
    "ph_forward_x": 40,
    "ph_forward_y": 41,
    "ph_forward_z": 42,

    # rel pressure
    "sw_rel_pres": 43
}
