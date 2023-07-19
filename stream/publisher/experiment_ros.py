import logging
import os.path
import threading
import time
from datetime import datetime
from enum import Enum

import torch
import queue
import numpy as np

import rospy
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import PoseStamped

from predict import estimate_joints
from data_types.bone_map import BoneMap
from utility import transformations
from pynput.keyboard import Listener


class exp_state(Enum):
    """
    States define what type of positions are sent to the robot
    """
    ROBOT_STOPPED = 1
    ROBOT_MOVING_TO_FOLLOW = 2
    ROBOT_FOLLOWING = 3
    ROBOT_MOVING_TO_TARGET = 4
    ROBOT_MOVING_TO_BIN = 5
    ROBOT_OPENING_GRIPPER = 6
    ROBOT_CLOSING_GRIPPER = 7


class commands(Enum):
    """
    States define what type of positions are sent to the robot
    """
    STOP = 1  # "stop!"
    FOLLOW_ME = 2  # "follow me"
    NEXT_TARGET = 3  # "go back"
    BIN = 4
    OPEN_GRIPPER = 5  # "open"
    CLOSE_GRIPPER = 6


task_targets = {
    "bin": np.array([0.4, 0.2, 0.2]),
    "A": np.array([0.5, 0.02, 0.4]),
    "B": np.array([0.3, 0.02, 0.4]),
    "C": np.array([0.5, 0.02, 0.5]),
    "D": np.array([0.3, 0.02, 0.5]),
    "E": np.array([0.5, 0.02, 0.6]),
    "F": np.array([0.3, 0.02, 0.6])
}


class ros_experiment_manager:

    def __init__(self, sensor_q: queue, key_q: queue, params: dict):
        self.__robot_pos = np.array([0.2579995474692755, 0.3034, 0.2864])  # current position of the robot end effector
        self.__target_list = [v for k, v in task_targets.items() if k != "bin"]
        self.__state = exp_state.ROBOT_STOPPED  # current state of the experiment
        self.__positions_queue = []  # the end effector will move along these positions in this list
        self.__gripper_max = 45
        self.__gripper_min = 0
        self.__gripper_state = self.__gripper_min  # degree to which the gripper is closed
        self.__active = False
        self.__sensor_q = sensor_q
        self.__key_q = key_q
        self.__larm_length = BoneMap.DEFAULT_LARM_LEN
        self.__uarm_length = BoneMap.DEFAULT_UARM_LEN

        self.__prev_follow_me = False

        # load the trained network
        torch.set_default_dtype(torch.float64)
        self.__device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # load model from given parameters
        self.__nn_model = nns.load_model_from_params(params=params)
        self.__y_targets = params["y_targets"]

        # check parameters
        if params["normalize"]:
            raise UserWarning("normalized data not supported")
        if params["dropout"] <= 0.0:
            raise UserWarning("a dropout is mandatory for MC predictions")

        # set sequence length only if a recurrent net is in use
        if params["model"] not in [nns_time.SplitModelLSTM, nns_time.DropoutLSTM]:
            self.__sequence_len = 1
        else:
            self.__sequence_len = params["sequence_len"]

        # ROS publisher
        self.__publisher = rospy.Publisher("/smartwatch_stream", Float32MultiArray, queue_size=1)
        self.__rate = rospy.Rate(ROS_RATE)  # hz

        # ROS Subscriber
        self.target_pose = PoseStamped()
        self.cube_pose = PoseStamped()

        rospy.Subscriber("/vrpn_client_node/Cube/pose", PoseStamped, self.cube_pose_callback)
        rospy.Subscriber("/vrpn_client_node/Target/pose", PoseStamped, self.target_pose_callback)

        # used to estimate delta time between observations
        self.__sw_prev_pred_t = None
        # used to stack observations for predictions with sequence len>1
        self.__sw_data_hist_seq = []
        # we also stack predictions for smoothing
        self.__sw_pred_times = []
        self.__sw_pred_hist = []

        # default hand and larm positions and rotations
        self.__pred_hand_origin_rua = np.array([0, 0, 0])
        self.__pred_larm_rot_rh = np.array([1, 0, 0, 0])
        self.__pred_larm_origin_rua = np.array([0, 0, 0])
        self.__pred_uarm_rot_rh = np.array([1, 0, 0, 0])

    def cube_pose_callback(self, msg):
        self.cube_pose = msg

    def target_pose_callback(self, msg):
        self.target_pose = msg

    def on_release(self, key):
        print('{0} release'.format(key))
        if hasattr(key, "char"):
            if key.char == 's':
                self.__key_q.put(commands.STOP.value)
            elif key.char == 'f':
                self.__key_q.put(commands.FOLLOW_ME.value)
            elif key.char == 'o':
                self.__key_q.put(commands.OPEN_GRIPPER.value)
            elif key.char == 'b':
                self.__key_q.put(commands.NEXT_TARGET.value)

    def start(self):
        update_thread = threading.Thread(target=self.update)
        update_thread.start()

        self.__active = True

        keyboard_thread = threading.Thread(target=self.listen_to_keyboard)
        keyboard_thread.start()

    def listen_to_keyboard(self):
        # Collect events until released
        with Listener(on_release=self.on_release) as listener:
            logging.info("listening to keyboard")
            listener.join()

    def create_movement_path(self,
                             start: np.array,
                             end: np.array,
                             start_up: bool = False,
                             end_up: bool = False):

        pos_list = []
        t_steps = float(ROS_RATE) * 3  # complete trajectory in 2 sec
        p_steps = float(ROS_RATE) * 1.5  # complete for start up or end up in 1 sec

        if start_up:
            for x in np.arange((p_steps + 1)):
                ss = x / float(p_steps)
                pos_list.append(start + np.array([0, ss * 0.2, 0]))
            start = pos_list[-1]

        end_list = []
        if end_up:
            end = end + np.array([0, 0.2, 0])
            for x in np.arange((p_steps + 1)):
                ss = x / float(p_steps)
                end_list.append(end - np.array([0, ss * 0.2, 0]))

        diff = end - start
        for x in np.arange(t_steps + 1):
            ss = x / float(t_steps)
            spos = ss * diff + start
            pos_list.append(spos)

        if end_up:
            pos_list += end_list
        return pos_list

    def implement_commands(self):
        command = None
        # read the latest speech recognition command
        while not self.__key_q.empty():
            command = self.__key_q.get()

        if command is None:
            return
        elif command == commands.STOP.value:
            self.__state = exp_state.ROBOT_STOPPED
            # clear any stored trajectory
            self.__positions_queue.clear()
            logging.info("Stopping all movement")
        elif command == commands.FOLLOW_ME.value:
            self.__state = exp_state.ROBOT_MOVING_TO_FOLLOW
            # overwrite positions queue with trajectory to smartwatch position
            sw_pos = self.get_predicted_robot_position()
            self.__positions_queue = self.create_movement_path(self.__robot_pos, sw_pos)
            logging.info("Moving towards smartwatch position")
        elif command == commands.NEXT_TARGET.value:
            self.__state = exp_state.ROBOT_MOVING_TO_TARGET
            if len(self.__target_list) > 0:
                # overwrite positions queue with trajectory to first task target
                self.__positions_queue = self.create_movement_path(self.__robot_pos, self.__target_list[0], end_up=True)
                logging.info("Moving to next target (targets {})".format(len(self.__target_list)))
            else:
                self.__key_q.put(commands.STOP.value)
                logging.info("No next target available")
        elif command == commands.BIN.value:
            self.__state = exp_state.ROBOT_MOVING_TO_BIN
            # overwrite positions queue with trajectory to first task target
            self.__positions_queue = self.create_movement_path(self.__robot_pos, task_targets["bin"], start_up=True)
            logging.info("Moving to bin")
        elif command == commands.OPEN_GRIPPER.value:
            self.__state = exp_state.ROBOT_OPENING_GRIPPER
            self.__positions_queue.clear()  # clear any stored trajectory
            logging.info("Opening gripper")
        elif command == commands.CLOSE_GRIPPER.value:
            self.__state = exp_state.ROBOT_CLOSING_GRIPPER
            self.__positions_queue.clear()  # clear any stored trajectory
            logging.info("Closing gripper")

        self.write_to_log(command)

    def get_predicted_robot_position(self):
        ee_pos = self.__pred_hand_origin_rua.copy()
        ee_pos[0] += 0.1
        ee_pos[1] += 0.25
        ee_pos[2] += 0.05
        return ee_pos

    def predict_from_smartwatch_data(self):
        """
        this function reads the last smartwatch messages and predicts positions and orientations from it\
        :return: pred_hand_origin_rua, pred_larm_origin_rua, pred_larm_rot_rh, pred_uarm_rot_rh
        """

    def update(self):
        logging.info("started update function")

        dat = 0
        while self.__active:
            # make a smartwatch prediction every step for the visualisation
            self.predict_from_smartwatch_data()

            # check the commands queue
            self.implement_commands()

            if self.__state == exp_state.ROBOT_FOLLOWING:
                # append the predicted hand position to the destination positions queue
                self.__positions_queue.append(self.get_predicted_robot_position())

            elif self.__state == exp_state.ROBOT_MOVING_TO_FOLLOW:
                if len(self.__positions_queue) == 0:
                    # the robot has arrived at the smartwatch position. Now follow
                    self.__state = exp_state.ROBOT_FOLLOWING
                    logging.info("Following smartwatch")

            elif self.__state == exp_state.ROBOT_MOVING_TO_BIN:
                if len(self.__positions_queue) == 0:
                    # the robot has arrived at the bin position. Now open gripper
                    self.__key_q.put(commands.OPEN_GRIPPER.value)

            elif self.__state == exp_state.ROBOT_MOVING_TO_TARGET:
                if len(self.__positions_queue) == 0:
                    # the robot has arrived at the target cube. Now close gripper
                    if len(self.__target_list) > 0:
                        if np.allclose(self.__robot_pos, self.__target_list[0]):
                            self.__target_list.pop(0)
                            if len(self.__target_list) == 0:
                                self.__target_list = [v for k, v in task_targets.items() if k != "bin"]
                            self.__key_q.put(commands.CLOSE_GRIPPER.value)

            elif self.__state == exp_state.ROBOT_OPENING_GRIPPER:
                if self.__gripper_state > self.__gripper_min:
                    self.__gripper_state -= 1
                else:
                    # the robot has opened the gripper
                    self.__gripper_state = self.__gripper_min

                    #  If it is above the bin, move to next target
                    if np.allclose(self.__robot_pos, task_targets["bin"]):
                        self.__key_q.put(commands.NEXT_TARGET.value)
                    else:
                        self.__key_q.put(commands.STOP.value)

            elif self.__state == exp_state.ROBOT_CLOSING_GRIPPER:
                if self.__gripper_state < self.__gripper_max:
                    self.__gripper_state += 1
                else:
                    # the robot has closed the gripper, move to bin
                    self.__gripper_state = self.__gripper_max
                    self.__key_q.put(commands.BIN.value)

            # update the position variable if a new position is available
            if len(self.__positions_queue) > 0:
                self.__robot_pos = self.__positions_queue.pop(0)

            # craft ros message and send
            msg = Float32MultiArray()
            # combine positions and rotations
            msg.data = np.hstack([
                self.__pred_hand_origin_rua,
                self.__pred_larm_rot_rh,
                self.__pred_larm_origin_rua,
                self.__pred_uarm_rot_rh,
                self.__robot_pos,
                self.__gripper_state
            ])
            self.__publisher.publish(msg)
            # sleep to not send messages too fast
            dat += 1
            self.__rate.sleep()
