import socket
import struct
import numpy as np

from wear_mocap_ape.data_types.bone_map import BoneMap
from wear_mocap_ape.estimate.watch_phone import WatchPhone


class WatchPhoneUDP(WatchPhone):
    def __init__(self,
                 ip: str,
                 port: int,
                 smooth: int = 5,
                 left_hand_mode=True,
                 tag: str = "PUB WATCH PHONE",
                 bonemap: BoneMap = None):
        super().__init__(smooth=smooth,
                         left_hand_mode=left_hand_mode,
                         tag=tag,
                         bonemap=bonemap)
        self.__ip = ip
        self.__port = port
        self.__tag = tag
        self.__port = port
        self.__udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def process_msg(self, msg: np.array):
        # craft UDP message and send
        msg = struct.pack('f' * len(msg), *msg)
        self.__udp_socket.sendto(msg, (self.__ip, self.__port))
