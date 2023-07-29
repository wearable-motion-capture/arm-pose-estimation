import logging
import queue
import threading
import wear_mocap_ape.config as config

from wear_mocap_ape.stream.publisher.watch_only_udp import WatchOnlyUDP
from wear_mocap_ape.stream.listener.imu import ImuListener
from wear_mocap_ape.data_types import messaging

logging.basicConfig(level=logging.INFO)

# adjust this to your local IP
ip = config.IP_OWN  # it should be a string, e.g., "192.168.1.101"

# listener and publisher run in separate threads. Listener fills the queue, publisher empties it
q = queue.Queue()

# the listener fills the que with received and parsed smartwatch data
imu_l = ImuListener(
    ip=ip,
    msg_size=messaging.watch_only_imu_msg_len,
    port=config.PORT_LISTEN_WATCH_IMU_LEFT  # the smartwatch app sends on this PORT
)
imu_w_l = threading.Thread(
    target=imu_l.listen,
    args=(q,)
)
imu_w_l.start()

# the publisher estimates arm poses from queued data and publishes them via UDP to given IP and PORT
joints_p = WatchOnlyUDP(
    ip=ip,
    port=config.PORT_PUB_WATCH_IMU_LEFT
)
udp_publisher = threading.Thread(
    target=joints_p.stream_loop,
    args=(q,)
)
udp_publisher.start()
