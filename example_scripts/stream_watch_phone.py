import logging
import queue
import threading

import wear_mocap_ape.config as config
from wear_mocap_ape.stream.listener.imu import ImuListener
from wear_mocap_ape.stream.publisher.watch_phone import WatchPhonePublisher
from wear_mocap_ape.utility import messaging

# enable basic logging
logging.basicConfig(level=logging.INFO)

# adjust this to your local IP
ip = config.IP_OWN  # it should be a string, e.g., "192.168.1.101"

# data processing happens in independent threads.
# We exchange data via queues.
# This script has two separate queues for the case that the
# user streams from a left-hand and a right-hand device at the same time
left_q = queue.Queue()  # data for left-hand mode
right_q = queue.Queue()  # data for right-hand mode

# left listener
imu_l = ImuListener(
    ip=config.IP_OWN,
    msg_size=messaging.watch_phone_imu_msg_len,
    port=config.PORT_LISTEN_WATCH_PHONE_IMU_LEFT,
    tag="LISTEN IMU LEFT"
)
l_thread = threading.Thread(
    target=imu_l.listen,
    args=(left_q,)
)
l_thread.start()

# right listener
imu_r = ImuListener(
    ip=config.IP_OWN,
    msg_size=messaging.watch_phone_imu_msg_len,
    port=config.PORT_LISTEN_WATCH_PHONE_IMU_RIGHT,
    tag="LISTEN IMU RIGHT"
)
r_thread = threading.Thread(
    target=imu_r.listen,
    args=(right_q,)
)
r_thread.start()

# left publisher
wp2ul = WatchPhonePublisher(
    ip=config.IP_OWN,
    port=config.PORT_PUB_WATCH_PHONE_LEFT,
    tag="PUBLISH LEFT"
)
ul_thread = threading.Thread(
    target=wp2ul.stream_loop,
    args=(left_q,)
)
ul_thread.start()

# right publisher
wp2ur = WatchPhonePublisher(
    ip=config.IP_OWN,
    port=config.PORT_PUB_WATCH_PHONE_RIGHT,
    tag="PUBLISH RIGHT",
    left_hand_mode=False
)
ur_thread = threading.Thread(
    target=wp2ur.stream_loop,
    args=(right_q,)
)
ur_thread.start()
