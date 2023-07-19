import logging
import queue
import threading
import config

from data_types.bone_map import BoneMap
from stream.listener.watch_imu import listen_for_watch_imu
from stream.publisher.watch_to_unity import WatchPublisher
from stream.listener.imu import ImuListener
from stream.publisher.watch_to_unity import WatchToUnity
from utility import deploy_models
from utility import messaging

logging.basicConfig(level=logging.INFO)

mode_params = deploy_models.FF.NORM_UARM_LARM.value
bonemap = BoneMap("Skeleton_21")

# listener and predictor run in separate threads. Listener fills the queue, predictor empties it
q = queue.Queue()

# the listener fills the que with transmitted smartwatch data
imu_l = ImuListener(
    msg_size=messaging.sw_standalone_imu_msg_len,
    port=config.WATCH_PORT
)
sensor_listener = threading.Thread(
    target=imu_l.listen(q),
    args=(q,)
)
sensor_listener.start()

# make predictions and stream them to Unity
w2u = WatchPublisher(bonemap=bonemap,
                     model_params=mode_params,
                     monte_carlo_samples=5,
                     smooth=25,
                     stream_monte_carlo=False)
udp_publisher = threading.Thread(
    target=w2u.stream_loop,
    args=(q,)
)
udp_publisher.start()
