import logging
import queue
import threading
from stream.listener.watch_and_phone_imu import listen_for_watch_and_phone_imu

from stream.publisher.watch_phone_to_unity import WatchPhoneToUnity

logging.basicConfig(level=logging.INFO)

# listener and predictor run in separate threads. Listener fills the queue, predictor empties it
sensor_que = queue.Queue()

# the listener fills the que with transmitted smartwatch and phone data
sensor_listener = threading.Thread(
    target=listen_for_watch_and_phone_imu,
    args=(sensor_que,)
)
sensor_listener.start()

# this thread broadcasts lower and upper arm orientations via UDP
wp2u = WatchPhoneToUnity()
udp_publisher = threading.Thread(
    target=wp2u.stream_loop,
    args=(sensor_que,)
)
udp_publisher.start()
