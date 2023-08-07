import socket
import struct

import wear_mocap_ape.config as config
from wear_mocap_ape.data_deploy.nn import deploy_models
from wear_mocap_ape.data_types.bone_map import BoneMap
from wear_mocap_ape.estimate.watch_only import WatchOnly


class WatchOnlyUDP(WatchOnly):
    def __init__(self,
                 ip: str,
                 port: int = config.PORT_PUB_LEFT_ARM,
                 model_hash: str = deploy_models.LSTM.ORI_CALIB_UARM_LARM.value,
                 smooth: int = 10,
                 stream_monte_carlo=True,
                 monte_carlo_samples=25,
                 bonemap: BoneMap = None,
                 tag: str = "PUB WATCH"):
        super().__init__(model_hash=model_hash, smooth=smooth,
                         stream_monte_carlo=stream_monte_carlo,
                         monte_carlo_samples=monte_carlo_samples,
                         bonemap=bonemap,
                         tag=tag)
        self.__tag = tag
        self.__port = port
        self.__ip = ip
        self.__udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def process_msg(self, msg: list):
        """
        The paren class calls this method
        whenever a new arm pose estimation finished
        """
        msg = struct.pack('f' * len(msg), *msg)
        self.__udp_socket.sendto(msg, (self.__ip, self.__port))
