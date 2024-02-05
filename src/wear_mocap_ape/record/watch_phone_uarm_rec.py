import datetime
from pathlib import Path
import numpy as np
from wear_mocap_ape.data_types.bone_map import BoneMap
from wear_mocap_ape.estimate.watch_phone_uarm import WatchPhoneUarm


class WatchPhoneUarmRecorder(WatchPhoneUarm):
    def __init__(self,
                 file: Path,
                 smooth: int = 5,
                 tag: str = "REC WATCH PHONE",
                 bonemap: BoneMap = None):
        super().__init__(smooth=smooth,
                         tag=tag,
                         bonemap=bonemap)
        self.__tag = tag
        self.__file = Path(file)
        self.__tag = tag

        # the header matches the msg in process_msg
        header = [
            "time",
            "hand_quat_w", "hand_quat_x", "hand_quat_y", "hand_quat_z",
            "hand_orig_rh_x", "hand_orig_rh_y", "hand_orig_rh_z",
            "larm_quat_rh_w", "larm_quat_rh_x", "larm_quat_rh_y", "larm_quat_rh_z",
            "larm_orig_rh_x", "larm_orig_rh_y", "larm_orig_rh_z",
            "uarm_quat_rh_w", "uarm_quat_rh_x", "uarm_quat_rh_y", "uarm_quat_rh_z",
            "uarm_orig_rh_x", "uarm_orig_rh_y", "uarm_orig_rh_z",
            "hips_quat_g_w", "hips_quat_g_x", "hips_quat_g_y", "hips_quat_g_z"
        ]

        if not self.__file.parent.exists():
            raise UserWarning(f"Directory does not exist {file.parent}")

        with open(self.__file, 'w') as fd:
            fd.write(",".join(header) + "\n")

    def process_msg(self, msg: np.array):
        """
        The paren class calls this method
        whenever a new arm pose estimation finished
        """
        msg = msg.tolist()
        with open(self.__file, 'a') as fd:
            msg.insert(0, datetime.datetime.now())
            msg = [str(x) for x in msg]
            fd.write(",".join(msg) + "\n")
