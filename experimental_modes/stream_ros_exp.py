import logging
import queue
import threading

import rospy

import config
from data_deploy.nn import deploy_models
from experimental_modes.ros_exp.ros_exp_manager import RosExperimentManager
from stream.listener.imu import ImuListener
from utility import messaging

# enable basic logging
logging.basicConfig(level=logging.INFO)

# init ros node to stream
rospy.init_node("smartwatch_stream")

imu_q = queue.Queue()
keyword_q = queue.Queue()
model_hash = deploy_models.FF.H_6DRR.value

# listen to watch data
imu_l = ImuListener(
    msg_size=messaging.watch_only_imu_msg_len,
    port=config.PORT_LISTEN_WATCH_PHONE_IMU_LEFT
)
imu_w_l = threading.Thread(
    target=imu_l.listen,
    args=(imu_q,)
)
imu_w_l.start()

# run the ros experiment
rem = RosExperimentManager(model_hash=model_hash, keyword_q=keyword_q)
rem.start(imu_q)
