import logging
import socket
import struct
import time
from datetime import datetime
from queue import Queue

import wear_mocap_ape.config as config


class AudioUDP:
    def __init__(self,
                 ip: str,
                 port: int = config.PORT_PUB_TRANSCRIBED_KEYS,
                 tag: str = "AUDIO PUB"):
        self.__ip = ip
        self.__port = port
        self.__tag = tag
        self.__socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.__socket.settimeout(5)

    def send_int_msg(self, msg: int) -> int:
        # craft UDP message and send
        return self.__socket.sendto(struct.pack('i', msg), (self.__ip, self.__port))

    def stream_loop(self, q: Queue):
        # used to estimate delta time and processing speed in Hz
        start = datetime.now()
        dat = 0

        logging.info("[{}] audio publish loop".format(self.__tag))
        # this loops while the socket is listening and/or receiving data
        while True:

            # get the most recent smartwatch data row from the queue
            row = q.get()
            while q.qsize() > 5:
                row = q.get()

            if row is not None:
                # second-wise updates to determine message frequency
                now = datetime.now()
                if (now - start).seconds >= 5:
                    start = now
                    logging.info("[{}] {} Hz".format(self.__tag, dat / 5))
                    dat = 0

                try:
                    # send message to Unity
                    self.send_int_msg(msg=row)
                except TimeoutError:
                    logging.info(f"[{self.__tag}] timed out")
                    continue
                dat += 1
                time.sleep(0.01)
