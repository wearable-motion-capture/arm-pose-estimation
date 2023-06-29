import logging
import queue
import threading
from stream.listener.dual_imu import dual_imu_listener

from stream.publisher.dual_orientations_quat_only import dual_orientations_quat_only

# from stream.publisher.dual_orientations import dual_orientations

logging.basicConfig(level=logging.INFO)

# listener and predictor run in separate threads. Listener fills the queue, predictor empties it
sensor_que = queue.Queue()

# the listener fills the que with transmitted smartwatch and phone data
sensor_listener = threading.Thread(
    target=dual_imu_listener,
    args=(sensor_que,)
)
sensor_listener.start()

# this listener fills the keyword_queue with transcribed mic data
# keywords are in utility.voice_commands import commands
# keyword_que = queue.Queue()
# keyword_trigger = threading.Thread(
#     target=voice_command_listener,
#     args=(keyword_que,)
# )
# keyword_trigger.start()

# this thread broadcasts lower and upper arm orientations via UDP
udp_publisher = threading.Thread(
    target=dual_orientations_quat_only,
    args=(sensor_que,)
)
udp_publisher.start()
