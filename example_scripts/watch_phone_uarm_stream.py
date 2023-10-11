import argparse
import atexit
import logging
import queue
import signal
import threading

import wear_mocap_ape.config as config
from wear_mocap_ape.stream.listener.imu import ImuListener
from wear_mocap_ape.stream.publisher.watch_phone_uarm_udp import WatchPhoneUarmUDP
from wear_mocap_ape.data_types import messaging

# enable basic logging
logging.basicConfig(level=logging.INFO)

# Instantiate the parser
parser = argparse.ArgumentParser(description='streams data from the watch in standalone mode')

# Required IP argument
parser.add_argument('ip', type=str,
                    help=f'put your local IP here. '
                         f'The script will publish arm '
                         f'pose data on PORT {config.PORT_PUB_LEFT_ARM} (left hand)'
                         f'or PORT {config.PORT_PUB_RIGHT_ARM} (right hand)')
args = parser.parse_args()
ip = args.ip

# data processing happens in independent threads.
# We exchange data via queues.
# This script has two separate queues for the case that the
# user streams from a left-hand and a right-hand device at the same time
left_q = queue.Queue()  # data for left-hand mode
right_q = queue.Queue()  # data for right-hand mode

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

# right listener
imu_r = ImuListener(
    ip=ip,
    msg_size=messaging.watch_phone_imu_msg_len,
    port=config.PORT_LISTEN_WATCH_PHONE_IMU_RIGHT,
    tag="LISTEN IMU RIGHT"
)
r_thread = threading.Thread(
    target=imu_r.listen,
    args=(right_q,)
)

# left publisher
wp2ul = WatchPhoneUarmUDP(
    ip=ip,
    port=config.PORT_PUB_LEFT_ARM,
    tag="PUBLISH LEFT"
)
ul_thread = threading.Thread(
    target=wp2ul.stream_loop,
    args=(left_q,)
)

# right publisher
wp2ur = WatchPhoneUarmUDP(
    ip=ip,
    port=config.PORT_PUB_RIGHT_ARM,
    tag="PUBLISH RIGHT",
    left_hand_mode=False
)
ur_thread = threading.Thread(
    target=wp2ur.stream_loop,
    args=(right_q,)
)

l_thread.start()
r_thread.start()
ul_thread.start()
ur_thread.start()


def terminate_all(*args):
    imu_l.terminate()
    wp2ur.terminate()
    imu_r.terminate()
    wp2ul.terminate()


# make sure all handler exit on termination
atexit.register(terminate_all)
signal.signal(signal.SIGTERM, terminate_all)
signal.signal(signal.SIGINT, terminate_all)
