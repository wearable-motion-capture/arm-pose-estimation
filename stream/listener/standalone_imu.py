import array
import logging
import socket
import queue
from datetime import datetime
import config
from utility.messaging import sw_standalone_imu_lookup

MSG_LEN = len(sw_standalone_imu_lookup) * 4
IP = config.IP
PORT = 46000
TAG = "SW UDP IMU"


def standalone_imu_listener(q: queue):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind((IP, PORT))
    logging.info("[{}] listening on {} - {}".format(TAG, IP, PORT))
    # begin receiving the data
    dat, start = 0, datetime.now()
    while 1:
        # log message frequency updates
        now = datetime.now()
        if (now - start).seconds >= 5:
            logging.info("[{}] {} Hz".format(TAG, dat / 5))
            dat, start = 0, now
        # receive and queue the data
        data, _ = s.recvfrom(MSG_LEN)
        if not data:
            logging.info("[{}] Stream closed".format(TAG))
            break
        else:
            adata = array.array('f', data)
            adata.byteswap()  # change endianness
            q.put(adata)
            dat += 1
