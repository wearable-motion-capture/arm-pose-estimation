import numpy as np
import rospy
from std_msgs.msg import Float32MultiArray

from wear_mocap_ape.data_types.bone_map import BoneMap
from wear_mocap_ape.estimate.watch_phone_uarm import WatchPhoneUarm


class WatchPhoneROS(WatchPhoneUarm):

    def __init__(self,
                 ros_node_name="/wear_mocap",
                 ros_rate=20,
                 smooth: int = 5,
                 tag: str = "PUB WATCH PHONE",
                 bonemap: BoneMap = None):
        super().__init__(smooth=smooth,
                         tag=tag,
                         bonemap=bonemap)

        # init ros node to stream
        self.__tag = tag
        rospy.init_node(ros_node_name, log_level=rospy.INFO)
        self.__publisher = rospy.Publisher(ros_node_name, Float32MultiArray, queue_size=1)
        self.__rate = rospy.Rate(ros_rate)
        rospy.loginfo(f"[{self.__tag}] Initiated ROS publisher")

    def process_msg(self, np_msg: np.array):
        """
        The paren class calls this method
        whenever a new arm pose estimation finished
        """
        # craft ros message and send
        msg = Float32MultiArray()
        # combine positions and rotations
        msg.data = np_msg
        self.__publisher.publish(msg)
        # sleep to not send messages too fast
        self.__rate.sleep()
