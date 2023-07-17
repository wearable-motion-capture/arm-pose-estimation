import logging
import queue
import threading

import config
from stream.listener.imu import ImuListener
from stream.publisher.watch_phone_to_unity import WatchPhoneToUnity
from utility import messaging

# enable basic logging
logging.basicConfig(level=logging.INFO)

# adjust IP to your needs
ip = config.IP

# data processing happens in independent threads.
# We exchange data via queues.
# This script has two separate queues for the case that the
# user streams from a left-hand and a right-hand device at the same time
left_q = queue.Queue()  # data for left-hand mode
right_q = queue.Queue()  # data for right-hand mode

# left listener
imu_l = ImuListener(
    ip=config.IP,
    msg_size=len(messaging.dual_imu_msg_lookup) * 4,
    port=config.WATCH_PHONE_PORT_LEFT,
    tag="IMU LEFT"
)
l_thread = threading.Thread(
    target=imu_l.listen,
    args=(left_q,)
)
l_thread.start()

# right listener
imu_r = ImuListener(
    ip=config.IP,
    msg_size=len(messaging.dual_imu_msg_lookup) * 4,
    port=config.WATCH_PHONE_PORT_RIGHT,
    tag="IMU RIGHT"
)
r_thread = threading.Thread(
    target=imu_r.listen,
    args=(right_q,)
)
r_thread.start()

# left publisher
wp2ul = WatchPhoneToUnity(
    ip=config.IP,
    port=config.UNITY_WATCH_PHONE_PORT_LEFT,
    tag="UNITY LEFT"
)
ul_thread = threading.Thread(
    target=wp2ul.stream_loop,
    args=(left_q,)
)
ul_thread.start()

# right publisher
wp2ur = WatchPhoneToUnity(
    ip=config.IP,
    port=config.UNITY_WATCH_PHONE_PORT_RIGHT,
    tag="UNITY RIGHT",
    left_hand_mode=False
)
ur_thread = threading.Thread(
    target=wp2ur.stream_loop,
    args=(right_q,)
)
ur_thread.start()
