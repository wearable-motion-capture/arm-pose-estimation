import logging
import queue
import threading

from data_types.bone_map import BoneMap
from stream.listener.standalone_imu import standalone_imu_listener
from stream.publisher.joints_adv import unity_stream_joints_adv
from utility import deploy_models

logging.basicConfig(level=logging.INFO)

prediction_params = deploy_models.FF.NORM_UARM_LARM.value
bonemap = BoneMap("Skeleton_21")

# listener and predictor run in separate threads. Listener fills the queue, predictor empties it
sensor_que = queue.Queue()

# the listener fills the que with transmitted smartwatch data
sensor_listener = threading.Thread(
    target=standalone_imu_listener,
    args=(sensor_que,)
)
sensor_listener.start()

# this thread makes predictions and sends them away to ros, which forwards them to unity
udp_publisher = threading.Thread(
    target=unity_stream_joints_adv,
    args=(sensor_que, bonemap, prediction_params, True)
)
udp_publisher.start()
