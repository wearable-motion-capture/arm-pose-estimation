import atexit
import logging
import queue
import signal
import threading
import wear_mocap_ape.config as config
from wear_mocap_ape.record.watch_phone_uarm_rec import WatchPhoneUarmRecorder

from wear_mocap_ape.stream.listener.imu import ImuListener
from wear_mocap_ape.data_types import messaging

import argparse

# Instantiate the parser
parser = argparse.ArgumentParser(description='records data from the watch in standalone mode')
# Required arguments
parser.add_argument('ip', type=str,
                    help=f'put your local IP here. '
                         f'The script will listen to watch data on '
                         f'PORT {config.PORT_LISTEN_WATCH_PHONE_IMU}')
parser.add_argument('file', type=str,
                    help=f'recorded data will be written into this file')
args = parser.parse_args()

# enable basic logging
logging.basicConfig(level=logging.INFO)

# data processing happens in independent threads.
# We exchange data via queues.
# This script has two separate queues for the case that the
# user streams from a left-hand and a right-hand device at the same time
left_q = queue.Queue()  # data for left-hand mode

# left listener
imu_l = ImuListener(
    ip=args.ip,
    msg_size=messaging.watch_phone_imu_msg_len,
    port=config.PORT_LISTEN_WATCH_PHONE_IMU,
    tag="LISTEN IMU LEFT"
)
l_thread = threading.Thread(
    target=imu_l.listen,
    args=(left_q,)
)

wp_rl = WatchPhoneUarmRecorder(file=args.file)
rl_thread = threading.Thread(
    target=wp_rl.processing_loop,
    args=(left_q,)
)

l_thread.start()
rl_thread.start()


def terminate_all(*args):
    imu_l.terminate()
    wp_rl.terminate()


# make sure all handler exit on termination
atexit.register(terminate_all)
signal.signal(signal.SIGTERM, terminate_all)
signal.signal(signal.SIGINT, terminate_all)
