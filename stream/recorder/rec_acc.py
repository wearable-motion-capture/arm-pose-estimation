import logging
import os
import random
import threading
from datetime import datetime
import queue

import numpy as np
from matplotlib import pyplot as plt, animation
from pynput.keyboard import Listener
import config
from utility import messaging, transformations

TAG = "REC ACC"


class imu_acc_recorder:
    def __init__(self, sensor_q: queue):
        self.__sensor_q = sensor_q
        fig, self.__ax = plt.subplots(nrows=3)

        self.__transform = True
        plt.suptitle(f"transform: {self.__transform}")

        self.__thist = []
        self.__xhist = []
        self.__yhist = []
        self.__zhist = []

        self.__line_x, = self.__ax[0].plot(self.__xhist)
        self.__ax[0].set_title("x acc")
        self.__line_y, = self.__ax[1].plot(self.__yhist)
        self.__ax[1].set_title("y acc")
        self.__line_z, = self.__ax[2].plot(self.__zhist)
        self.__ax[2].set_title("z acc")

        self.__hist_len = 300
        self.__start = datetime.now()
        self.__step = 0

        ani = animation.FuncAnimation(fig, self.animate, interval=0.1)

        self.__slp = messaging.sw_standalone_imu_lookup
        plt.tight_layout()
        plt.show()

    def animate(self, i):

        # get the most recent smartwatch data row from the queue
        row = None
        while not self.__sensor_q.empty():
            row = self.__sensor_q.get()

        # process received data
        if row is not None:
            sec = (datetime.now() - self.__start).total_seconds()
            lacc = [
                row[self.__slp["lacc_x"]],
                row[self.__slp["lacc_y"]],
                row[self.__slp["lacc_z"]]
            ]

            if self.__transform:
                sw_rot = transformations.sw_quat_to_global(np.array([
                    row[self.__slp["rotvec_w"]],
                    row[self.__slp["rotvec_x"]],
                    row[self.__slp["rotvec_y"]],
                    row[self.__slp["rotvec_z"]]
                ]))

                # quaternion to rotate smartwatch coord y-axis towards north
                north_rad = -row[self.__slp["north_deg"]] * 0.01745329
                north_quat = transformations.euler_to_quat(np.array([0, north_rad, 0], dtype=np.float64))
                # now align to known North Pole position. We have our global rotation
                sw_rot_quat_rh = transformations.hamilton_product(north_quat, sw_rot)

                # linear acceleration relative to the hip
                lacc_g = transformations.sw_pos_to_global(np.array(lacc))
                sw_inv_rot = transformations.quat_invert(sw_rot_quat_rh)
                lacc_rh = transformations.quat_rotate_vector(sw_inv_rot, lacc_g)
                lacc = lacc_rh

            self.__xhist.append(lacc[0])
            self.__yhist.append(lacc[1])
            self.__zhist.append(lacc[2])

            self.__thist.append(sec)
            while len(self.__yhist) > self.__hist_len:
                del self.__xhist[0]
                del self.__yhist[0]
                del self.__zhist[0]
                del self.__thist[0]

            self.__line_x.set_data(self.__thist, self.__xhist)
            self.__line_y.set_data(self.__thist, self.__yhist)
            self.__line_z.set_data(self.__thist, self.__zhist)

            for ax in self.__ax:
                ax.relim()
                ax.set_ylim((-10, 10))
                ax.autoscale_view()

        return self.__line_x, self.__line_y, self.__line_z
