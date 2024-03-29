import argparse
import queue
import threading

import wear_mocap_ape.config as config
from wear_mocap_ape.stream.listener.imu import ImuListener
from wear_mocap_ape.stream.publisher.watch_phone_uarm_ros import WatchPhoneROS
from wear_mocap_ape.data_types import messaging

# parse command line arguments
parser = argparse.ArgumentParser(description='streams microphone data from the watch in standalone mode, '
                                             'transcribes it, and checks for a list of keywords for voice commands.')
# Required IP argument
parser.add_argument('ip', type=str, help=f'put your local IP here.')
args = parser.parse_args()
ip = args.ip

left_q = queue.Queue()  # data for left-hand mode

# left listener
imu_l = ImuListener(
    ip=ip,
    msg_size=messaging.watch_phone_imu_msg_len,
    port=config.PORT_LISTEN_WATCH_PHONE_IMU_LEFT,
    tag="LISTEN IMU LEFT"
)
l_thread = threading.Thread(
    target=imu_l.listen,
    args=(left_q,)
)
l_thread.start()

# left ROS publisher
wp2ul = WatchPhoneROS()
ul_thread = threading.Thread(
    target=wp2ul.processing_loop,
    args=(left_q,)
)
ul_thread.start()
