import datetime
import queue

import numpy as np
from matplotlib import pyplot as plt, animation
from utility import messaging, transformations
from utility.transformations import integrate_series, moving_average

TAG = "ACC Ball Vis"


def plot_acc_segment(sensor_q: queue, standalone_mode: bool):
    if standalone_mode:
        slp = messaging.sw_standalone_imu_lookup
    else:
        slp = messaging.dual_imu_msg_lookup

    thist = []
    xl = []
    yl = []
    zl = []

    start = None
    while len(thist) < 200:
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
        lacc = [
            row[slp["sw_lacc_x"]],
            row[slp["sw_lacc_y"]],
            row[slp["sw_lacc_z"]]
        ]

        thist.append(t)

        xl.append(lacc[0])
        yl.append(lacc[1])
        zl.append(lacc[2])

    w = 5
    xl = moving_average(xl, w)
    xv = integrate_series(0, xl[1:], thist)
    xv = moving_average(xv, w)
    xp = integrate_series(0, xv[1:], thist)

    yl = moving_average(yl, w)
    yv = integrate_series(0, yl[1:], thist)
    yv = moving_average(yv, w)
    yp = integrate_series(0, yv[1:], thist)

    zl = moving_average(zl, w)
    zv = integrate_series(0, zl[1:], thist)
    zv = moving_average(zv, w)
    zp = integrate_series(0, zv[1:], thist)

    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(15, 8), sharey='col')

    ax[0, 0].plot(thist, xl)
    ax[0, 0].set_title("x acc")
    ax[0, 1].plot(thist, xp)
    ax[0, 1].set_title("DDx")
    ax[1, 0].plot(thist, yl)
    ax[1, 0].set_title("y acc")
    ax[1, 1].plot(thist, yp)
    ax[1, 1].set_title("DDy")
    ax[2, 0].plot(thist, zl)
    ax[2, 0].set_title("z acc")
    ax[2, 1].plot(thist, zp)
    ax[2, 1].set_title("DDz")

    plt.tight_layout()
    plt.show()
