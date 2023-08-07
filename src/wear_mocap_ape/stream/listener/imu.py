import array
import logging
import socket
import queue
from datetime import datetime


class ImuListener:
    def __init__(self,
                 msg_size: int,
                 port: int,
                 ip: str,
                 tag: str = "IMU LISTENER"):
        """
        :param msg_size: the message size the listener should expect
        :param port: the port on which to listen for the messages
        :param ip: the own local IP
        :param tag: this tag will be prepended to logger messages and printouts
        """

        self.__active = False
        self.__msg_size = msg_size
        self.__ip = ip
        self.__port = port
        self.__tag = tag

    def terminate(self):
        self.__active = False

    def listen(self, q: queue):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.bind((self.__ip, self.__port))
        logging.info(f"[{self.__tag}] listening on {self.__ip} - {self.__port}")

        # begin receiving the data
        dat, start = 0, datetime.now()
        self.__active = True
        while self.__active:
            # log message frequency updates
            now = datetime.now()
            if (now - start).seconds >= 5:
                logging.info(f"[{self.__tag}] {dat / 5} Hz")
                dat, start = 0, now

            # receive and queue the data
            data, _ = s.recvfrom(self.__msg_size)
            if not data:
                logging.info(f"[{self.__tag}] Stream closed")
                break
            else:
                adata = array.array('f', data)
                adata.byteswap()  # change endianness
                q.put(adata)
                dat += 1
