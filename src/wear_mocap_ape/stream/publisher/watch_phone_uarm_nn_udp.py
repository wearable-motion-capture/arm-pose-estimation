import logging
import socket
import struct
import numpy as np

from wear_mocap_ape.data_deploy.nn.deploy_models import LSTM
from wear_mocap_ape.data_types.bone_map import BoneMap
from wear_mocap_ape.estimate.watch_phone_uarm_nn import WatchPhoneUarmNN


class WatchPhoneUarmNnUDP(WatchPhoneUarmNN):
    def __init__(self,
                 ip: str,
                 port: int,
                 smooth: int = 1,
                 model_hash: str = LSTM.WATCH_PHONE_UARM.value,
                 stream_mc: bool = True,
                 tag: str = "PUB WATCH PHONE",
                 bonemap: BoneMap = None):
        super().__init__(
            smooth=smooth,
            tag=tag,
            bonemap=bonemap,
            stream_mc=stream_mc,
            model_hash=model_hash
        )
        self.__ip = ip
        self.__port = port
        self.__tag = tag
        self.__port = port
        self.__udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.__udp_socket.settimeout(5)

    def process_msg(self, msg: np.array):
        """
        The paren class calls this method
        whenever a new arm pose estimation finished
        """
        # craft UDP message and send
        msg = struct.pack('f' * len(msg), *msg)
        try:
            self.__udp_socket.sendto(msg, (self.__ip, self.__port))
        except TimeoutError:
            logging.info(f"[{self.__tag}] timed out")
