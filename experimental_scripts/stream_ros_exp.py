import queue
import threading

import wear_mocap_ape.config as config
from wear_mocap_ape.stream.listener.imu import ImuListener
from wear_mocap_ape.stream.publisher.watch_phone_to_ros import WatchPhoneToROS
from wear_mocap_ape.utility import messaging

# adjust this to your local IP
ip = config.IP_OWN  # it should be a string, e.g., "192.168.1.101"

left_q = queue.Queue()  # data for left-hand mode

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

# left ROS publisher
wp2ul = WatchPhoneToROS()
ul_thread = threading.Thread(
    target=wp2ul.stream_loop,
    args=(left_q,)
)
ul_thread.start()
