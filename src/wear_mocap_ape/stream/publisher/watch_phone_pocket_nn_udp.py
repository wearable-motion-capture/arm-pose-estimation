import socket
import struct

import numpy as np

from wear_mocap_ape.data_deploy.nn import deploy_models
from wear_mocap_ape.data_types.bone_map import BoneMap
from wear_mocap_ape.estimate.watch_phone_pocket_nn import WatchPhonePocketNN


class WatchPhonePocketNnUDP(WatchPhonePocketNN):
    def __init__(self,
                 ip,
                 port,
                 smooth: int = 1,
                 model_hash: str = deploy_models.LSTM.WATCH_PHONE_POCKET.value,
                 stream_mc: bool = True,
                 mc_samples: int = 25,
                 bonemap: BoneMap = None,
                 tag: str = "PHONE POCKET NN UDP"):
        super().__init__(
            smooth=smooth,
            model_hash=model_hash,
            stream_monte_carlo=stream_mc,
            monte_carlo_samples=mc_samples,
            bonemap=bonemap
        )

        self.__tag = tag
        self.__port = port
        self.__ip = ip
        self.__udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.__udp_socket.settimeout(5)

    def process_msg(self, msg: np.array) -> int:
        # craft UDP message and send
        msg = struct.pack('f' * len(msg), *msg)
        return self.__udp_socket.sendto(msg, (self.__ip, self.__port))
