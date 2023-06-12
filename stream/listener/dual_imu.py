import array
import logging
import socket
import queue

from datetime import datetime
from enum import Enum

import config
from utility.messaging import dual_imu_msg_lookup


class SensorStream(Enum):
    CHUNK_SIZE = len(dual_imu_msg_lookup) * 4
    IP = config.IP
    PORT = 65000
    TAG = "DUAL IMU"


def dual_imu_listener(q: queue):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(
        (SensorStream.IP.value,
         SensorStream.PORT.value)
    )

    logging.info(
        "[{}] listening on {} - {}".format(SensorStream.TAG.value,
                                           SensorStream.IP.value,
                                           SensorStream.PORT.value)
    )
    msg_len = len(dual_imu_msg_lookup)

    # begin receiving the data
    start = datetime.now()
    dat = 0
    while 1:
        # second-wise updates to determine message frequency
        now = datetime.now()
        if (now - start).seconds >= 5:
            start = now
            logging.info("[{}] {} Hz".format(SensorStream.TAG.value, dat / 5))
            dat = 0

        # this function waits
        data, _ = s.recvfrom(SensorStream.CHUNK_SIZE.value)

        if not data:
            logging.info("[{}] Stream closed".format(SensorStream.TAG.value))
            break
        else:

            adata = array.array('f', data)
            adata.byteswap()  # change endianness

            if len(adata) != msg_len:
                logging.info(
                    "Skipped message because it "
                    "contained {}/{} entries".format(len(adata), msg_len)
                )
                continue

            # the data comes as a csv string encapsulated by START and END
            q.put(adata)
            dat += 1
