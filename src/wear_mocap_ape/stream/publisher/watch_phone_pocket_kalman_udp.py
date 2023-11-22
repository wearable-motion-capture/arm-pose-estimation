import logging
import socket
import struct
import numpy as np
from wear_mocap_ape.estimate.watch_phone_pocket_kalman import WatchPhonePocketKalman


class WatchPhonePocketKalmanUDP(WatchPhonePocketKalman):
    def __init__(self,
                 ip,
                 port,
                 smooth: int = 1,
                 num_ensemble: int = 32,
                 model_name="SW-model-sept-4",
                 window_size: int = 10,
                 stream_mc: bool = True,
                 tag: str = "KALMAN UDP POCKET PHONE"):

        super().__init__(
            smooth=smooth,
            num_ensemble=num_ensemble,
            model_name=model_name,
            window_size=window_size,
            stream_mc=stream_mc
        )
        self.__tag = tag
        self.__port = port
        self.__ip = ip
        self.__udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.__udp_socket.settimeout(5)

    def process_msg(self, msg: np.array) -> int:
        # craft UDP message and send
        msg = struct.pack('f' * len(msg), *msg)
        try:
            return self.__udp_socket.sendto(msg, (self.__ip, self.__port))
        except TimeoutError:
            logging.info(f"[{self.__tag}] timed out")
            return -1
