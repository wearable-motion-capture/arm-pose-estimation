# enable basic logging
import argparse
import logging
import queue
import threading

from wear_mocap_ape import config
from wear_mocap_ape.data_deploy.nn.deploy_models import LSTM
from wear_mocap_ape.data_types import messaging
from wear_mocap_ape.stream.listener.imu import ImuListener
from wear_mocap_ape.stream.publisher.free_hips_udp import FreeHipsUDP

logging.basicConfig(level=logging.INFO)

# Instantiate the parser
parser = argparse.ArgumentParser(description='')

# Required IP argument
parser.add_argument('ip', type=str, help=f'put your local IP here.')
args = parser.parse_args()
ip = args.ip

model_hash = LSTM.FREE_HIPS_REC.value

# data for left-hand mode
left_q = queue.Queue()

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

# left publisher
fhp = FreeHipsUDP(ip=ip,
                  model_hash=model_hash,
                  port=config.PORT_PUB_LEFT_ARM)
p_thread = threading.Thread(
    target=fhp.stream_loop,
    args=(left_q,)
)
p_thread.start()
