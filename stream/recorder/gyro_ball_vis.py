import datetime
import queue

import numpy as np
from matplotlib import pyplot as plt, animation
from utility import messaging, transformations
from utility.transformations import integrate_series, moving_average

TAG = "ACC Ball Vis"


def plot_gyro_segment(sensor_q: queue, standalone_mode: bool):
    if standalone_mode:
        slp = messaging.sw_standalone_imu_lookup
    else:
        slp = messaging.dual_imu_msg_lookup

    thist = []
    xv = []
    yv = []
    zv = []

    start = None
    while len(thist) < 500:
        # get the most recent smartwatch data row from the queue
        row = sensor_q.get()

        t = datetime.datetime(
            year=1970,
            day=1,
            month=1,
            hour=int(row[slp["sw_h"]]),
            minute=int(row[slp["sw_m"]]),
            second=int(row[slp["sw_s"]]),
            microsecond=int(row[slp["sw_ns"]] * 0.001)  # nanoseconds to microseconds
        ).timestamp()

        if start == None:
            start = t
            t = 0
        else:
            t = t - start

        dt = row[slp["sw_dt"]]
        gyro = [
            row[slp["sw_gyro_x"]],
            row[slp["sw_gyro_y"]],
            row[slp["sw_gyro_z"]]
        ]

        thist.append(t)

        xv.append(gyro[0])
        yv.append(gyro[1])
        zv.append(gyro[2])

    xp = integrate_series(0, xv[1:], thist)
    yp = integrate_series(0, yv[1:], thist)
    zp = integrate_series(0, zv[1:], thist)

    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(15, 8), sharey='col')

    ax[0, 0].plot(thist, xv)
    ax[0, 0].set_title("x gyro")
    ax[0, 1].plot(thist, xp)
    ax[0, 1].set_title("Dx")
    ax[1, 0].plot(thist, yv)
    ax[1, 0].set_title("y gyro")
    ax[1, 1].plot(thist, yp)
    ax[1, 1].set_title("Dy")
    ax[2, 0].plot(thist, zv)
    ax[2, 0].set_title("z gyro")
    ax[2, 1].plot(thist, zp)
    ax[2, 1].set_title("Dz")

    plt.tight_layout()
    plt.show()
