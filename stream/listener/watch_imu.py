import array
import logging
import socket
import queue
from datetime import datetime
import config
from utility.messaging import sw_standalone_imu_lookup


def listen_for_watch_imu(q: queue):
    msg_size = len(sw_standalone_imu_lookup) * 4
    ip = config.IP
    port = 46000
    log_tag = "SW UDP IMU"

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind((ip, port))
    logging.info("[{}] listening on {} - {}".format(log_tag, ip, port))
    # begin receiving the data
    dat, start = 0, datetime.now()
    while 1:
        # log message frequency updates
        now = datetime.now()
        if (now - start).seconds >= 5:
            logging.info("[{}] {} Hz".format(log_tag, dat / 5))
            dat, start = 0, now
        # receive and queue the data
        data, _ = s.recvfrom(msg_size)
        if not data:
            logging.info("[{}] Stream closed".format(log_tag))
            break
        else:
            adata = array.array('f', data)
            adata.byteswap()  # change endianness
            q.put(adata)
            dat += 1
