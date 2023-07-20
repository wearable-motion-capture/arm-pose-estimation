import logging
import queue
import threading

import rospy

import config
from stream.listener.imu import ImuListener
from stream.publisher.watch_phone import WatchPhonePublisher
from utility import messaging

# enable basic logging
logging.basicConfig(level=logging.INFO)

# adjust IP to your needs
ip = config.IP

# init ros node to stream
rospy.init_node("smartwatch_stream")

# data processing happens in independent threads.
# We exchange data via queues.
# This script has two separate queues for the case that the
# user streams from a left-hand and a right-hand device at the same time
left_q = queue.Queue()  # data for left-hand mode
right_q = queue.Queue()  # data for right-hand mode

# left listener
imu_l = ImuListener(
    ip=config.IP,
    msg_size=len(messaging.WATCH_PHONE_IMU_LOOKUP) * 4,
    port=config.LISTEN_WATCH_PHONE_IMU_LEFT,
    tag="IMU LEFT"
)
l_thread = threading.Thread(
    target=imu_l.listen,
    args=(left_q,)
)
l_thread.start()

# left publisher
wp2ul = WatchPhonePublisher(
    ip=config.IP,
    port=config.PUB_WATCH_PHONE_LEFT,
    tag="UNITY LEFT"
)
ul_thread = threading.Thread(
    target=wp2ul.stream_loop,
    args=(left_q,)
)
ul_thread.start()
