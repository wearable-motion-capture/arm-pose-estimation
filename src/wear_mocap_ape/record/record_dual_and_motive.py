import logging
import queue
import threading

from wear_mocap_ape.config import PORT_LISTEN_WATCH_PHONE_IMU_LEFT, PORT_PUB_WATCH_PHONE_LEFT
from wear_mocap_ape.record.file_writer.watch_phone_motive_to_csv import watch_phone_motive_to_csv
from wear_mocap_ape.stream.listener.imu import ImuListener
from wear_mocap_ape.stream.listener.motive_q import MotiveQListener
from wear_mocap_ape.stream.publisher.watch_phone import WatchPhonePublisher
from wear_mocap_ape.stream.publisher.motive_to_unity import MotiveToUnity
from wear_mocap_ape.utility import messaging

logging.basicConfig(level=logging.INFO)

# the listener fills the queue with transmitted smartwatch and phone data
sensor_que = queue.Queue()
imu_l = ImuListener(msg_size=messaging.watch_phone_imu_msg_len, port=PORT_LISTEN_WATCH_PHONE_IMU_LEFT)
sensor_listener = threading.Thread(target=imu_l.listen, args=(sensor_que,))
sensor_listener.start()
# the publisher processes received data and publishes it to other services, e.g., Unity
imu_p = WatchPhonePublisher(port=PORT_PUB_WATCH_PHONE_LEFT, smooth=0)

# this listener keeps track of motive ground truth data
motive_listener = MotiveQListener()
update_thread = threading.Thread(target=motive_listener.stream_loop)
update_thread.start()
# these publishers help to send debug info to our Unity visualization
motive_publisher = MotiveToUnity()

# finally, this function records data to a file and feed the debug visualization
watch_phone_motive_to_csv(
    sensor_q=sensor_que,
    motive_listener=motive_listener,
    debug_motive_publisher=motive_publisher,
    dual_publisher=imu_p
)
