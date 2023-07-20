import logging
import queue
import threading
import config

from stream.publisher.watch import WatchPublisher
from stream.listener.imu import ImuListener
from utility import deploy_models
from utility import messaging

logging.basicConfig(level=logging.INFO)

model_params = deploy_models.FF.H_XYZ.value

# listener and predictor run in separate threads. Listener fills the queue, predictor empties it
q = queue.Queue()

# the listener fills the que with transmitted smartwatch data
imu_l = ImuListener(
    msg_size=messaging.watch_only_imu_msg_len,
    port=config.PORT_LISTEN_WATCH_IMU_LEFT
)
sensor_listener = threading.Thread(
    target=imu_l.listen,
    args=(q,)
)
sensor_listener.start()

# make predictions and stream them to Unity
w2u = WatchPublisher(
    model_params=model_params,
    monte_carlo_samples=5,
    smooth=25,
    stream_monte_carlo=False
)
udp_publisher = threading.Thread(
    target=w2u.stream_loop,
    args=(q,)
)
udp_publisher.start()
