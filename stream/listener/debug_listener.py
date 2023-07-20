import array
import logging
import socket
import queue
from datetime import datetime
import config
from utility.messaging import WATCH_ONLY_IMU_LOOKUP



msg_size = 72
ip = config.IP
port = config.UNITY_WATCH_PHONE_PORT_LEFT

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.bind((ip, port))
# begin receiving the data
while 1:
    # receive and queue the data
    data, _ = s.recvfrom(msg_size)
    if not data:
        break
    else:
        adata = array.array('f', data)
        # adata.byteswap()  # change endianness
        print(adata[4], adata[5], adata[6])
