import atexit
import logging
import queue
import signal
import threading
from datetime import datetime
from pathlib import Path

import wear_mocap_ape.config as config
from wear_mocap_ape.record.est_output import EstOutputRecorder

from wear_mocap_ape.stream.listener.imu import ImuListener
from wear_mocap_ape.data_types import messaging

import argparse

# Instantiate the parser
parser = argparse.ArgumentParser(description='records data from the watch in standalone mode')
# Required arguments
parser.add_argument('ip', type=str,
                    help=f'put your local IP here. '
                         f'The script will listen to watch data on '
                         f'PORT {config.PORT_LISTEN_WATCH_IMU}')
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
    port=config.PORT_LISTEN_WATCH_IMU  # the smartwatch app sends on this PORT
)
imu_w_l = threading.Thread(
    target=imu_l.listen,
    args=(q,)
)

watch_rec = EstOutputRecorder(file=args.file)

prev_time = datetime.now()
active = True
file = Path(args.file)

header = [
    "timestamp",
    "sw_dt",
    "sw_h",
    "sw_m",
    "sw_s",
    "sw_ns",
    "sw_rotvec_w",
    "sw_rotvec_x",
    "sw_rotvec_y",
    "sw_rotvec_z",
    "sw_rotvec_conf",
    "sw_gyro_x",
    "sw_gyro_y",
    "sw_gyro_z",
    "sw_lvel_x",
    "sw_lvel_y",
    "sw_lvel_z",
    "sw_lacc_x",
    "sw_lacc_y",
    "sw_lacc_z",
    "sw_pres",
    "sw_grav_x",
    "sw_grav_y",
    "sw_grav_z",
    "sw_forward_w",
    "sw_forward_x",
    "sw_forward_y",
    "sw_forward_z",
    "sw_init_pres"
]

if not file.parent.exists():
    raise UserWarning(f"Directory does not exist {file.parent}")

with open(file, 'w') as fd:
    fd.write(",".join(header) + "\n")


def processing_loop(sensor_q: queue = None):
    logging.info(f"[RAW_RECORDER] starting loop")

    # used to estimate delta time and processing speed in Hz
    start = datetime.now()
    dat = 0

    # this loops while the socket is listening and/or receiving data
    while active:

        # processing speed output
        now = datetime.now()
        if (now - start).seconds >= 5:
            start = now
            logging.info(f"[RAW_RECORDER] {dat / 5} Hz")
            dat = 0

        # get the most recent smartwatch data row from the queue
        row = sensor_q.get()
        while sensor_q.qsize() > 5:
            row = sensor_q.get()

        msg = list(row)
        with open(file, 'a') as fd:
            msg.insert(0, datetime.now())
            msg = [str(x) for x in msg]
            fd.write(",".join(msg) + "\n")
        dat += 1


recorder = threading.Thread(
    target=processing_loop,
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
signal.signal(signal.SIGINT, terminate_all)
