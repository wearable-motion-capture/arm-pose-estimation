import logging
import queue
import threading

from record.file_writer.watch_phone_motive_to_csv import watch_phone_motive_to_csv
from stream.listener.motive import MotiveListener
from stream.listener.watch_and_phone_imu import listen_for_watch_and_phone_imu
from stream.publisher.watch_phone_to_unity import WatchPhoneToUnity
from stream.publisher.motive_to_unity import MotiveToUnity

# start ros node
logging.basicConfig(level=logging.INFO)

# listener and predictor run in separate threads. Listener fills the queue, predictor empties it
sensor_que = queue.Queue()

# the listener fills the que with transmitted smartwatch and phone data
sensor_listener = threading.Thread(
    target=listen_for_watch_and_phone_imu,
    args=(sensor_que,)
)
sensor_listener.start()

# this listener keeps track of motive ground truth data
motive_listener = MotiveListener()
update_thread = threading.Thread(target=motive_listener.stream_loop)
update_thread.start()

# these publishers help to send debug info to our Unity visualization
motive_publisher = MotiveToUnity()
dual_publisher = WatchPhoneToUnity(smooth=0)

# finally, this function records data to a file and feed the debug visualization
watch_phone_motive_to_csv(
    sensor_q=sensor_que,
    motive_listener=motive_listener,
    debug_motive_publisher=motive_publisher,
    debug_dual_publisher=dual_publisher
)
