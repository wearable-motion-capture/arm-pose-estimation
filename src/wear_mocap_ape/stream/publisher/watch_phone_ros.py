from datetime import datetime
import queue
import numpy as np
import rospy
from std_msgs.msg import Float32MultiArray

from wear_mocap_ape.data_types.bone_map import BoneMap
from wear_mocap_ape.utility import transformations as ts
from wear_mocap_ape.data_types import messaging


class WatchPhoneROS:

    def __init__(self,
                 ros_node_name="/wear_mocap",
                 ros_rate=20,
                 smooth: int = 5,
                 left_hand_mode=True,
                 tag: str = "PUB WATCH PHONE",
                 bonemap: BoneMap = None):

        # init ros node to stream
        self.__tag = tag
        self.__left_hand_mode = left_hand_mode
        rospy.init_node(ros_node_name, log_level=rospy.INFO)
        self.__publisher = rospy.Publisher(ros_node_name, Float32MultiArray, queue_size=1)
        self.__rate = rospy.Rate(ros_rate)
        rospy.loginfo(f"[{self.__tag}] Initiated ROS publisher")

        # average over multiple time steps
        self.__smooth = smooth
        self.__smooth_hist = []

        # simple lookup for values of interest
        self.__slp = messaging.WATCH_PHONE_IMU_LOOKUP

        # use body measurements for transitions
        if bonemap is None:
            self.__larm_vec = np.array([BoneMap.DEFAULT_LARM_LEN, 0, 0])
            self.__uarm_vec = np.array([BoneMap.DEFAULT_UARM_LEN, 0, 0])
        else:
            # get values from bone map
            self.__larm_vec = np.array([bonemap.left_lower_arm_length, 0, 0])
            self.__uarm_vec = np.array([bonemap.left_upper_arm_length, 0, 0])

        # in left hand mode, the arm is stretched along the negative X-axis in T pose
        if left_hand_mode:
            self.__larm_vec[0] = -self.__larm_vec[0]
            self.__uarm_vec[0] = -self.__uarm_vec[0]

    def calibrate_imus_with_offset(self, row: np.array):

        # get relevant entries from the row
        sw_fwd = np.array([
            row[self.__slp["sw_forward_w"]], row[self.__slp["sw_forward_x"]],
            row[self.__slp["sw_forward_y"]], row[self.__slp["sw_forward_z"]]
        ])

        sw_rot = np.array([
            row[self.__slp["sw_rotvec_w"]], row[self.__slp["sw_rotvec_x"]],
            row[self.__slp["sw_rotvec_y"]], row[self.__slp["sw_rotvec_z"]]
        ])

        ph_fwd = np.array([
            row[self.__slp["ph_forward_w"]], row[self.__slp["ph_forward_x"]],
            row[self.__slp["ph_forward_y"]], row[self.__slp["ph_forward_z"]]
        ])

        ph_rot = np.array([
            row[self.__slp["ph_rotvec_w"]], row[self.__slp["ph_rotvec_x"]],
            row[self.__slp["ph_rotvec_y"]], row[self.__slp["ph_rotvec_z"]]
        ])

        # estimate north quat to align Z-axis of global and android coord system
        if self.__left_hand_mode:
            quat_north = ts.calib_watch_left_to_north_quat(sw_fwd)
            # the arm orientations if the calib position with left arm forward is perfect
            larm_dst_g = np.array([-0.7071068, 0, -0.7071068, 0])
            uarm_dst_g = np.array([0.7071068, 0, 0.7071068, 0])

        else:
            quat_north = ts.calib_watch_right_to_north_quat(sw_fwd)
            # the arm orientation if calib position with right arm forward is perfect
            larm_dst_g = np.array([-0.7071068, 0.0, 0.7071068, 0.0])
            uarm_dst_g = np.array([0.7071068, 0.0, -0.7071068, 0.0])

        # calibrate watch with offset to perfect position
        sw_rot_g = ts.android_quat_to_global(sw_rot, quat_north)
        sw_fwd_g = ts.android_quat_to_global(sw_fwd, quat_north)
        sw_off_g = ts.hamilton_product(ts.quat_invert(sw_fwd_g), larm_dst_g)
        sw_cal_g = ts.hamilton_product(sw_rot_g, sw_off_g)

        # calibrate phone with offset to perfect position
        ph_rot_g = ts.android_quat_to_global(ph_rot, quat_north)
        ph_fwd_g = ts.android_quat_to_global(ph_fwd, quat_north)
        ph_off_g = ts.hamilton_product(ts.quat_invert(ph_fwd_g), uarm_dst_g)
        ph_cal_g = ts.hamilton_product(ph_rot_g, ph_off_g)

        return sw_cal_g, ph_cal_g

    def row_to_arm_pose(self, row):

        larm_quat, uarm_quat = self.calibrate_imus_with_offset(row)

        # store rotations in history if smoothing is required
        if self.__smooth > 1:
            self.__smooth_hist.append(np.hstack([larm_quat, uarm_quat]))
            if len(self.__smooth_hist) < self.__smooth:
                return None
            while len(self.__smooth_hist) > self.__smooth:
                del self.__smooth_hist[0]

            all_rots = np.vstack(self.__smooth_hist)
            # Calculate the mean of all predictions mean
            avg_larm_rot_r = ts.average_quaternions(all_rots[:, :4])
            avg_uarm_rot_r = ts.average_quaternions(all_rots[:, 4:])
        else:
            avg_larm_rot_r = larm_quat
            avg_uarm_rot_r = uarm_quat

        # get the transition from upper arm origin to lower arm origin
        # get transitions from lower arm origin to hand
        larm_origin_rua = ts.quat_rotate_vector(avg_uarm_rot_r, self.__uarm_vec)
        larm_rotated = ts.quat_rotate_vector(avg_larm_rot_r, self.__larm_vec)

        hand_origin_rua = larm_rotated + larm_origin_rua
        # this is the list for the actual joint positions and rotations
        return list(
            np.hstack([
                avg_larm_rot_r,  # in this case, hand rotation is larm rotation
                hand_origin_rua,
                avg_larm_rot_r,
                larm_origin_rua,
                avg_uarm_rot_r
            ])
        )

    def stream_loop(self, sensor_q: queue):
        # used to estimate delta time and processing speed in Hz
        start = datetime.now()
        dat = 0
        rospy.loginfo(f"[{self.__tag}] starting ROS stream loop")

        # this loops while the socket is listening and/or receiving data
        while True:
            # get the most recent smartwatch data row from the queue
            row = sensor_q.get()

            while sensor_q.qsize() > 5:
                row = sensor_q.get()

            # process received data
            if row is not None:
                # second-wise updates to determine message frequency
                now = datetime.now()
                if (now - start).seconds >= 5:
                    start = now
                    rospy.loginfo(f"[{self.__tag}] {self.__tag, dat / 5} Hz")
                    dat = 0

                # get message as numpy array
                np_msg = self.row_to_arm_pose(row)
                # can return None if not enough historical data for smoothing is available
                if np_msg is None:
                    continue

                # craft ros message and send
                msg = Float32MultiArray()
                # combine positions and rotations
                msg.data = np_msg
                self.__publisher.publish(msg)
                # sleep to not send messages too fast
                dat += 1
                self.__rate.sleep()
