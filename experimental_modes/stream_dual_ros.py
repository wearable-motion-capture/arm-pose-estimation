import logging
import queue
import threading

import rospy

from predict.models import PPGOneHot1dCNN
from stream.listener.dual_audio import dual_audio_listener
from stream.listener.dual_imu import dual_imu_listener
from stream.listener.dual_ppg import dual_ppg_listener
from stream.publisher.dual_joints_and_hand_action_ros import dual_orientations_and_hand_action_ros

logging.basicConfig(level=logging.INFO)

# the listener fills the que with transmitted smartwatch and phone data
imu_que = queue.Queue()
imu_listener = threading.Thread(target=dual_imu_listener, args=(imu_que,))
imu_listener.start()

# PPG data from the watch comes in a lower frequency.
# We catch it in a separate thread
ppg_que = queue.Queue()
ppg_listener = threading.Thread(target=dual_ppg_listener, args=(ppg_que,))
ppg_listener.start()

# this listener fills the keyword_queue with transcribed mic data
# keywords are in utility.voice_commands import commands
#
# The thread uses the Google API to transcribe the mic stream. Add
# `GOOGLE_APPLICATION_CREDENTIALS=/path/to/google_auth.json` to your environment variables before running the script.
# Set voice command keyword IDs in `utility/voice_commands.py`.
key_que = queue.Queue()
# keyword_trigger = threading.Thread(
#     target=dual_audio_listener,
#     args=(key_que,)
# )
# keyword_trigger.start()


ppg_ps = {
    "model": PPGOneHot1dCNN,
    "batch_size": 5,
    "learning_rate": 0.001,
    "epochs": 200,
    "concise": True
}

# this thread broadcasts everything as a ros topic
# order of the ros message:
# [
# hand_pos [x,y,z],
# larm_rot [w,x,y,z],
# elbow_pos [x,y,z],
# uarm_rot [w,x,y,z],
# PPG action [0. or 1.],
# transcribed keyword ID [ID] (make sure to enable the thread above for voice transcription)
# ]
rospy.init_node("smartwatch_stream")
udp_publisher = threading.Thread(
    target=dual_orientations_and_hand_action_ros,
    args=(imu_que, ppg_que, key_que, ppg_ps,)
)
udp_publisher.start()
