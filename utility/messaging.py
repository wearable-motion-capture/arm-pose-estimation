"""
Sensor values from the smartwatch come in a fixed order.
This lookup returns the position of specific measurements.
"""
sw_standalone_imu_lookup = {
    # timestamp
    "h": 0,  # hour
    "m": 1,  # minute
    "s": 2,  # second
    "ns": 3,  # nanosecond

    # orientation
    "rotvec_w": 4,
    "rotvec_x": 5,
    "rotvec_y": 6,
    "rotvec_z": 7,

    # gyro
    "gyro_x": 8,
    "gyro_y": 9,
    "gyro_z": 10,

    # velocity from lacc (1x integrated)
    "lvel_x": 11,
    "lvel_y": 12,
    "lvel_z": 13,

    # linear acceleration
    "lacc_x": 14,
    "lacc_y": 15,
    "lacc_z": 16,

    # atmospheric pressure
    "pres": 17,

    # gravity
    "grav_x": 18,
    "grav_y": 19,
    "grav_z": 20,

    # calibration measurements
    "init_pres": 21,  # pressure at chest height
    "north_deg": 22  # forward facing direction
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

    # gyro
    "sw_gyro_x": 8,
    "sw_gyro_y": 9,
    "sw_gyro_z": 10,

    # velocity from lacc (1x integrated)
    "sw_lvel_x": 11,
    "sw_lvel_y": 12,
    "sw_lvel_z": 13,

    # linear acceleration
    "sw_lacc_x": 14,
    "sw_lacc_y": 15,
    "sw_lacc_z": 16,

    # atmospheric pressure
    "sw_pres": 17,

    # gravity
    "sw_grav_x": 18,
    "sw_grav_y": 19,
    "sw_grav_z": 20,

    # phone data
    # timestamp
    "ph_h": 21,  # hour
    "ph_m": 22,  # minute
    "ph_s": 23,  # second
    "ph_ns": 24,  # nanosecond

    # orientation
    "ph_rotvec_w": 25,
    "ph_rotvec_x": 26,
    "ph_rotvec_y": 27,
    "ph_rotvec_z": 28,

    # gyro
    "ph_gyro_x": 29,
    "ph_gyro_y": 30,
    "ph_gyro_z": 31,

    # velocity from lacc (1x integrated)
    "ph_lvel_x": 32,
    "ph_lvel_y": 33,
    "ph_lvel_z": 34,

    # linear acceleration
    "ph_lacc_x": 35,
    "ph_lacc_y": 36,
    "ph_lacc_z": 37,

    # gravity
    "ph_grav_x": 38,
    "ph_grav_y": 39,
    "ph_grav_z": 40,

    # atmospheric pressure
    "ph_pres": 41,

    # calibration data
    # smartwatch forward
    "sw_forward_w": 42,
    "sw_forward_x": 43,
    "sw_forward_y": 44,
    "sw_forward_z": 45,

    # phone forward
    "ph_forward_w": 46,
    "ph_forward_x": 47,
    "ph_forward_y": 48,
    "ph_forward_z": 49,

    # rel pressure
    "sw_rel_pres": 50
}
