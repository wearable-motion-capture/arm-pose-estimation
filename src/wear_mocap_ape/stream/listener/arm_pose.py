import wear_mocap_ape.config as config
from wear_mocap_ape.stream.listener.imu import ImuListener


class ArmPoseListener(ImuListener):
    """
    Same structure as ImuListener, but we know the message size due to normed message composition
    """

    def __init__(
            self,
            port: int,
            ip: str = config.IP_OWN,
            tag: str = "ARM POSE LISTENER"
    ):
        super().__init__(
            msg_size=18 * 4,  # hand_quat, hand_orig, larm_quat, larm_orig, uarm_quat
            port=port,
            ip=ip,
            tag=tag
        )
