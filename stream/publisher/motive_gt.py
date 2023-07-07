import logging
import socket
import struct
import time
from datetime import datetime

import config
from stream.listener.motive import MotiveListener

TAG = "MotivePublisher"
IP = config.IP
PORT = 50003


def publish_gt(listener: MotiveListener):
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # used to estimate delta time and processing speed in Hz
    start = datetime.now()
    dat = 0

    logging.info("[{}] starting Unity stream loop".format(TAG))
    # this loops while the socket is listening and/or receiving data
    while True:
        # second-wise updates to determine message frequency
        now = datetime.now()
        if (now - start).seconds >= 5:
            start = now
            logging.info("[{}] {} Hz".format(TAG, dat / 5))
            dat = 0

        gt_message = listener.get_ground_truth()

        # don't send message if mocap lost track
        if gt_message is None:
            continue

        # craft UDP message and send
        msg = struct.pack('f' * len(gt_message), *gt_message)

        udp_socket.sendto(msg, (IP, PORT))
        dat += 1
        time.sleep(0.01)
