import logging
import threading
from enum import Enum
import queue
import numpy as np
import rospy
from std_msgs.msg import Float32MultiArray
import config
from stream.listener.arm_pose import ArmPoseListener
from stream.publisher.watch import WatchPublisher
from pynput.keyboard import Listener


class ROBOT_STATE(Enum):
    STOPPED = 1
    FOLLOWING = 3
    MOVING_TO_TARGET = 4
    SORTING_TARGET = 5
    OPENING_GRIPPER = 6
    CLOSING_GRIPPER = 7


class VOICE_COMMANDS(Enum):
    STOP = 3  # "stop"
    FOLLOW_ME = 5  # "follow"
    CONTINUE = 4  # "continue"
    OPEN_GRIPPER = 6  # "open"


red_bin = np.array([0.4, 0.2, 0.2])
grn_bin = np.array([0.4, 0.2, 0.2])

TARGETS = {
    "A": (np.array([0.5, 0.02, 0.4]), red_bin),
    "B": (np.array([0.3, 0.02, 0.4]), grn_bin),
    "C": (np.array([0.5, 0.02, 0.5]), grn_bin),
    "D": (np.array([0.3, 0.02, 0.5]), red_bin),
    "E": (np.array([0.5, 0.02, 0.6]), grn_bin),
    "F": (np.array([0.3, 0.02, 0.6]), red_bin)
}


class RosExperimentManager:
    ROS_RATE = 20  # hz
    MAX_SPEED = 1  # max displacement in "move towards" mode
    # degree to which the gripper is closed
    GRIPPER_MAX = 45
    GRIPPER_MIN = 0

    def __init__(self, model_hash: str, keyword_q):
        self.__keyword_q = keyword_q
        self.__target_list = [k for k in TARGETS.keys()]

        # this will make watch position predictions and publish them to the given PORT
        self.__ape = WatchPublisher(model_hash, port=config.PORT_PUB_ROS_EXP)
        self.__apl = ArmPoseListener(port=config.PORT_PUB_ROS_EXP)
        self.__apq = queue.Queue()  # store arm poses in here

        # these values describe the current robot state
        self.__robot_pos = np.array([0.2579995474692755, 0.3034, 0.2864])  # current position of the robot end effector
        self.__gripper_state = self.GRIPPER_MIN  # degree to which the gripper is closed

        # current state of the experiment
        self.__state = ROBOT_STATE.STOPPED
        self.__positions_queue = []  # the end effector will move along these positions in this list

        # ROS publisher
        self.__publisher = rospy.Publisher("/smartwatch_stream", Float32MultiArray, queue_size=1)
        self.__rate = rospy.Rate(self.ROS_RATE)

    def start(self, watch_imu_q: queue):
        """
        starts all the threads and the internal update loop
        """
        # arm pose estimation thread
        ape_thread = threading.Thread(target=self.__ape.stream_loop, args=(watch_imu_q,))
        ape_thread.start()
        # arm pose listener thread
        apl_thread = threading.Thread(target=self.__apl.listen, args=(self.__apq,))
        apl_thread.start()

        # listen for keyboard inputs
        keyboard_thread = threading.Thread(target=self.listen_to_keyboard)
        keyboard_thread.start()

        # run the experiment logic
        self.__state = ROBOT_STATE.STOPPED
        update_thread = threading.Thread(target=self.update)
        update_thread.start()

    def create_smooth_trajectory(self,
                                 start: np.array,
                                 end: np.array,
                                 start_up: bool = False,
                                 end_up: bool = False):

        pos_list = []
        t_steps = float(self.ROS_RATE) * 3  # complete trajectory in 2 sec
        p_steps = float(self.ROS_RATE) * 1.5  # complete for start up or end up in 1 sec

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

    def sort_target(self, target_pos):
        self.__state = ROBOT_STATE.SORTING_TARGET
        self.__positions_queue = self.create_smooth_trajectory(
            self.__robot_pos,
            target_pos,
            start_up=True
        )
        logging.info(f"Moving to bin at {target_pos}")

    def stop(self):
        self.__state = ROBOT_STATE.STOPPED
        self.__positions_queue.clear()  # clear any stored trajectory
        logging.info("Stopping all movement")

    def follow(self):
        self.__state = ROBOT_STATE.FOLLOWING
        self.__positions_queue.clear()  # clear any stored trajectory
        logging.info("Following smartwatch position")

    def next_target(self):
        self.__state = ROBOT_STATE.MOVING_TO_TARGET
        if len(self.__target_list) > 0:
            # overwrite positions queue with trajectory to first task target
            self.__positions_queue = self.create_smooth_trajectory(
                self.__robot_pos,
                TARGETS[self.__target_list[0]][0],
                end_up=True
            )
            logging.info(f"Moving to target {self.__target_list[0]} (targets {len(self.__target_list)})")
        else:
            # stop if no target remains
            self.__keyword_q.put(VOICE_COMMANDS.STOP.value)
            logging.info("No next target available")

    def open_gripper(self):
        self.__state = ROBOT_STATE.OPENING_GRIPPER
        self.__positions_queue.clear()  # clear any stored trajectory
        logging.info("Opening gripper")

    def close_gripper(self):
        self.__state = ROBOT_STATE.CLOSING_GRIPPER
        self.__positions_queue.clear()  # clear any stored trajectory
        logging.info("Closing gripper")

    def parse_voice_commands(self):
        keyword = None

        # read the latest speech recognition command
        while not self.__keyword_q.empty():
            keyword = self.__keyword_q.get()

        if keyword is None:
            return
        elif keyword == VOICE_COMMANDS.STOP.value:
            self.stop()
        elif keyword == VOICE_COMMANDS.FOLLOW_ME.value:
            self.follow()
        elif keyword == VOICE_COMMANDS.CONTINUE.value:
            self.next_target()
        elif keyword == VOICE_COMMANDS.OPEN_GRIPPER.value:
            self.open_gripper()

    def estimate_ee_position(self):
        # get the most recent arm pose estimation
        arm_pose = None
        while not self.__apq.empty():
            arm_pose = self.__apq.get()
        if arm_pose is not None:
            # we are only interested in the hand position
            ee_pos = arm_pose[4:7]
            # apply offset
            ee_pos[0] += 0.1
            ee_pos[1] += 0.25
            ee_pos[2] += 0.05
            return ee_pos
        else:
            return np.zeros(3)

    def move_towards(self, target):
        # speed limit expressed as vector magnitude
        max_m = self.MAX_SPEED * 1.0 / self.ROS_RATE

        # direction as difference between positions
        pos = self.__robot_pos
        diff = target - pos

        # check if magnitude of difference exceeds speed limit
        if np.linalg.norm(diff) > max_m:
            n_diff = diff / np.linalg.norm(diff)
            return n_diff * max_m
        else:
            return diff

    def update(self):
        dat = 0
        while True:

            # estimate at every step to not fill up the queue too much
            ee_pos = self.estimate_ee_position()

            # check the keywords queue for actions to take
            self.parse_voice_commands()

            if self.__state == ROBOT_STATE.FOLLOWING:
                # append the predicted hand position to the destination positions queue
                self.__positions_queue.append(self.move_towards(ee_pos))

            elif self.__state == ROBOT_STATE.SORTING_TARGET:
                # check if the robot has arrived at the bin position
                if len(self.__positions_queue) == 0:
                    # Now open gripper
                    self.open_gripper()

            elif self.__state == ROBOT_STATE.MOVING_TO_TARGET:
                # the robot has arrived at the target cube.
                if len(self.__positions_queue) == 0:
                    # Now close gripper
                    self.close_gripper()

            elif self.__state == ROBOT_STATE.OPENING_GRIPPER:
                if self.__gripper_state > self.GRIPPER_MIN:
                    self.__gripper_state -= 1
                else:
                    # the robot has opened the gripper
                    self.__gripper_state = self.GRIPPER_MIN

                    # #  If it is above the bin, move to next target
                    # if np.allclose(self.__robot_pos, task_targets["bin"]):
                    #     self.__action_q.put(actions.NEXT_TARGET.value)
                    # else:
                    #     self.__action_q.put(actions.STOP.value)

            elif self.__state == ROBOT_STATE.CLOSING_GRIPPER:
                if self.__gripper_state < self.GRIPPER_MAX:
                    self.__gripper_state += 1
                else:
                    # the robot has closed the gripper
                    self.__gripper_state = self.GRIPPER_MAX
                    # move to bin
                    if len(self.__target_list) > 0:
                        t_loc, t_bin = TARGETS[self.__target_list[0]]
                        if np.allclose(self.__robot_pos, t_loc):
                            self.__target_list.pop(0)
                            self.sort_target(t_bin)

            # update the position variable if a new position is available
            if len(self.__positions_queue) > 0:
                self.__robot_pos = self.__positions_queue.pop(0)

            # craft ros message and send
            msg = Float32MultiArray()
            # combine positions and rotations
            msg.data = np.hstack([
                self.__robot_pos,
                self.__gripper_state
            ])
            self.__publisher.publish(msg)
            # sleep to not send messages too fast
            dat += 1
            self.__rate.sleep()

    def on_release(self, key):
        print('{0} release'.format(key))
        if hasattr(key, "char"):
            if key.char == 's':  # stop
                self.__keyword_q.put(VOICE_COMMANDS.STOP.value)
            elif key.char == 'f':  # follow me
                self.__keyword_q.put(VOICE_COMMANDS.FOLLOW_ME.value)
            elif key.char == 'o':  # open
                self.__keyword_q.put(VOICE_COMMANDS.OPEN_GRIPPER.value)
            elif key.char == 'c':  # continue
                self.__keyword_q.put(VOICE_COMMANDS.CONTINUE.value)

    def listen_to_keyboard(self):
        # Collect events until released
        with Listener(on_release=self.on_release) as listener:
            logging.info("listening to keyboard")
            listener.join()
