import threading
from enum import Enum
import queue
import numpy as np
import rospy
from std_msgs.msg import Float32MultiArray
import config
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
grn_bin = np.array([-0.4, 0.2, 0.2])

TARGETS = {
    "A": (np.array([-0.1, 0.01, 0.42]), red_bin),
    "B": (np.array([0.1, 0.01, 0.42]), grn_bin),
    "C": (np.array([-0.1, 0.01, 0.52]), red_bin),
    "D": (np.array([0.1, 0.01, 0.52]), red_bin),
    "E": (np.array([-0.1, 0.01, 0.62]), red_bin),
    "F": (np.array([0.1, 0.01, 0.62]), grn_bin)
}


class WatchListener:
    pass


class RosExperimentManager:
    ROS_RATE = 20  # hz
    MAX_SPEED = 0.25  # max displacement in "move towards" mode
    # degree to which the gripper is closed
    GRIPPER_MAX = 45  # 45
    GRIPPER_MIN = 0
    HOME_POS = np.array([0.2579995474692755, 0.3034, 0.2864])

    def __init__(self, model_hash: str, keyword_q):
        self.__keyword_q = keyword_q
        self.__target_list = [k for k in TARGETS.keys()]

        # this will make watch position predictions and publish them to the given PORT
        self.__ape = WatchPublisher(
            model_hash,
            smooth=25,
            monte_carlo_samples=20,
            port=config.PORT_PUB_WATCH_PHONE_LEFT,
            stream_monte_carlo=False
        )

        # these values describe the current robot state
        self.__robot_pos = self.HOME_POS  # current position of the robot end effector
        self.__gripper_state = self.GRIPPER_MIN  # degree to which the gripper is closed
        self.__last_follow = False

        # current state of the experiment
        self.__state = ROBOT_STATE.STOPPED
        self.__positions_queue = []  # the end effector will move along these positions in this list

        # ROS publisher
        self.__publisher = rospy.Publisher("/smartwatch_stream", Float32MultiArray, queue_size=1)
        self.__rate = rospy.Rate(self.ROS_RATE)
        rospy.loginfo("Initiated Ros Experiment Manager")

    def start(self, watch_imu_q: queue):
        """
        starts all the threads and the internal update loop
        """
        # arm pose estimation thread
        ape_thread = threading.Thread(target=self.__ape.stream_loop, args=(watch_imu_q,))
        ape_thread.start()

        # listen for keyboard inputs
        keyboard_thread = threading.Thread(target=self.listen_to_keyboard)
        keyboard_thread.start()

        # run the experiment logic
        self.__state = ROBOT_STATE.STOPPED
        update_thread = threading.Thread(target=self.update)
        update_thread.start()
        rospy.loginfo("Started all threads")

    def create_smooth_trajectory(self,
                                 start: np.array,
                                 end: np.array,
                                 start_up: bool = False,
                                 end_up: bool = False):

        pos_list = []
        t_steps = float(self.ROS_RATE) * 4  # complete trajectory in 2 sec
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
        rospy.loginfo(f"Moving to bin at {target_pos}")

    def stop(self):
        self.__state = ROBOT_STATE.STOPPED
        self.__positions_queue.clear()  # clear any stored trajectory
        rospy.loginfo("Stopping all movement")

    def follow(self):
        self.__state = ROBOT_STATE.FOLLOWING
        self.__positions_queue.clear()  # clear any stored trajectory
        rospy.loginfo("Following smartwatch position")

    def next_target(self):
        self.__state = ROBOT_STATE.MOVING_TO_TARGET
        if len(self.__target_list) > 0:
            # overwrite positions queue with trajectory to first task target
            self.__positions_queue = self.create_smooth_trajectory(
                self.__robot_pos,
                TARGETS[self.__target_list[0]][0],
                end_up=True
            )
            rospy.loginfo(f"Moving to target {self.__target_list[0]} (targets {len(self.__target_list)})")
        else:
            # stop if no target remains
            self.__keyword_q.put(VOICE_COMMANDS.STOP.value)
            rospy.loginfo("No next target available")

    def open_gripper(self):
        self.__state = ROBOT_STATE.OPENING_GRIPPER
        self.__positions_queue.clear()  # clear any stored trajectory
        rospy.loginfo("Opening gripper")

    def close_gripper(self):
        self.__state = ROBOT_STATE.CLOSING_GRIPPER
        self.__positions_queue.clear()  # clear any stored trajectory
        rospy.loginfo("Closing gripper")

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
        arm_pose = self.__ape.get_last_msg()
        if arm_pose is not None:
            # we are only interested in the hand position
            ee_pos = arm_pose[4:7]
            # apply offset
            ee_pos[0] += 0.05
            ee_pos[1] += 0.25
            ee_pos[2] += 0.05
            return ee_pos
        else:
            return self.HOME_POS

    def move_towards(self, target):
        # speed limit expressed as vector magnitude
        max_m = self.MAX_SPEED * 1.0 / self.ROS_RATE

        # direction as difference between positions
        pos = self.__robot_pos
        diff = target - pos

        # check if magnitude of difference exceeds speed limit
        if np.linalg.norm(diff) > max_m:
            n_diff = diff / np.linalg.norm(diff)
            return pos + n_diff * max_m
        else:
            return pos + diff

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
                self.__last_follow = True

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
                    if self.__last_follow:
                        self.stop()
                        self.__last_follow = False
                    else:
                        self.next_target()

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
            rospy.loginfo("listening to keyboard")
            listener.join()
