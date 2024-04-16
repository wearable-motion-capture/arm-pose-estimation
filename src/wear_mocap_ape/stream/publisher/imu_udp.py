import logging
import queue
import socket
import struct
import threading

import wear_mocap_ape.config as config


class IMUPublisherUDP:
    def __init__(self,
                 ip: str,
                 port: int = config.PORT_PUB_LEFT_ARM,
                 tag: str = "PUB IMU"):

        self.__tag = tag
        self.__active = False
        self.__port = port
        self.__ip = ip
        self.__udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.__udp_socket.settimeout(5)

    def terminate(self):
        self.__active = False

    def publish_in_thread(self, q: queue):
        imu_w_l = threading.Thread(
            target=self.publish_loop,
            args=(q,)
        )
        imu_w_l.start()

    def publish_loop(self, q: queue):

        self.__active = True
        while self.__active:

            try:
                # get the most recent smartwatch data row from the queue
                msg = q.get(timeout=2)
                while q.qsize() > 5:
                    msg = q.get(timeout=2)
            except queue.Empty:
                logging.info(f"[{self.__tag}] no data")
                continue

            msg = struct.pack('f' * len(msg), *msg)
            try:
                self.__udp_socket.sendto(msg, (self.__ip, self.__port))
            except TimeoutError:
                logging.info(f"[{self.__tag}] send timed out")
