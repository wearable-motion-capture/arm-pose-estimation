import logging
import queue
import threading
import wear_mocap_ape.config as config

from wear_mocap_ape.stream.publisher.watch import WatchOnlyPublisher
from wear_mocap_ape.stream.listener.imu import ImuListener
from wear_mocap_ape.utility import messaging

logging.basicConfig(level=logging.INFO)

# listener and predictor run in separate threads. Listener fills the queue, predictor empties it
q = queue.Queue()

# the listener fills the que with transmitted smartwatch data
imu_l = ImuListener(
    msg_size=messaging.watch_only_imu_msg_len,
    port=config.PORT_LISTEN_WATCH_IMU_LEFT
)
imu_w_l = threading.Thread(
    target=imu_l.listen,
    args=(q,)
)
imu_w_l.start()

# make predictions and stream them to Unity
w2u = WatchOnlyPublisher()
udp_publisher = threading.Thread(
    target=w2u.stream_loop,
    args=(q,)
)
udp_publisher.start()
