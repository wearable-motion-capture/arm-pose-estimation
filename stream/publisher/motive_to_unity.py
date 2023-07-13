import logging
import socket
import struct
import time
from datetime import datetime

import numpy as np

import config
from stream.listener.motive import MotiveListener


class MotiveToUnity:
    def __init__(self):
        self.__tag = "MOTIVE TO UNITY"
        self.__ip = config.IP
        self.__port = config.UNITY_GT_PORT
        # UDP socket
        self.__socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send_np_msg(self, msg: np.array) -> int:
        # craft UDP message and send
        msg = struct.pack('f' * len(msg), *msg)
        return self.__socket.sendto(msg, (self.__ip, self.__port))

    def stream_loop(self, listener: MotiveListener):
        # used to estimate delta time and processing speed in Hz
        start = datetime.now()
        dat = 0

        logging.info("[{}] starting Unity stream loop".format(self.__tag))
        # this loops while the socket is listening and/or receiving data
        while True:
            # second-wise updates to determine message frequency
            now = datetime.now()
            if (now - start).seconds >= 5:
                start = now
                logging.info("[{}] {} Hz".format(self.__tag, dat / 5))
                dat = 0

            np_msg = listener.get_ground_truth()

            # don't send message if mocap lost track
            if np_msg is None:
                continue

            # send message to Unity
            self.send_np_msg(msg=np_msg)
            dat += 1
            time.sleep(0.01)
