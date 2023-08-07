import atexit
import logging
import queue
import signal
import threading
import wear_mocap_ape.config as config
from wear_mocap_ape.record.watch_only_rec import WatchOnlyRecorder

from wear_mocap_ape.stream.listener.imu import ImuListener
from wear_mocap_ape.data_types import messaging

import argparse

# Instantiate the parser
parser = argparse.ArgumentParser(description='records data from the watch in standalone mode')
# Required arguments
parser.add_argument('ip', type=str,
                    help=f'put your local IP here. '
                         f'The script will listen to watch data on '
                         f'PORT {config.PORT_LISTEN_WATCH_IMU_LEFT}')
parser.add_argument('file', type=str,
                    help=f'recorded data will be written into this file')
args = parser.parse_args()

logging.basicConfig(level=logging.INFO)

# listener and publisher run in separate threads. Listener fills the queue, publisher empties it
q = queue.Queue()

# the listener fills the que with received and parsed smartwatch data
imu_l = ImuListener(
    ip=args.ip,
    msg_size=messaging.watch_only_imu_msg_len,
    port=config.PORT_LISTEN_WATCH_IMU_LEFT  # the smartwatch app sends on this PORT
)
imu_w_l = threading.Thread(
    target=imu_l.listen,
    args=(q,)
)

watch_rec = WatchOnlyRecorder(file=args.file)
recorder = threading.Thread(
    target=watch_rec.processing_loop,
    args=(q,)
)

imu_w_l.start()
recorder.start()


def terminate_all(*args):
    imu_l.terminate()
    watch_rec.terminate()

# make sure all handler exit on termination
atexit.register(terminate_all)
signal.signal(signal.SIGTERM, terminate_all)
signal.signal(signal.SIGKILL, terminate_all)
signal.signal(signal.SIGINT, terminate_all)
