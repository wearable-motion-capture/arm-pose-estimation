import logging
import queue
import threading

from stream.listener.dual_imu import dual_imu_listener
from stream.listener.standalone_imu import standalone_imu_listener
from stream.recorder.rec_acc import imu_acc_recorder

if __name__ == "__main__":
    # start ros node
    logging.basicConfig(level=logging.INFO)

    # listener and predictor run in separate threads. Listener fills the queue, predictor empties it
    sensor_que = queue.Queue()

    standalone_mode = False

    if standalone_mode:
        # the listener fills the que with transmitted smartwatch data
        sensor_listener = threading.Thread(
            target=standalone_imu_listener,
            args=(sensor_que,)
        )
        sensor_listener.start()

        recorder = imu_acc_recorder(sensor_q=sensor_que, standalone_mode=standalone_mode)
    else:
        # the listener fills the que with transmitted smartwatch data
        sensor_listener = threading.Thread(
            target=dual_imu_listener,
            args=(sensor_que,)
        )
        sensor_listener.start()

        recorder = imu_acc_recorder(sensor_q=sensor_que, standalone_mode=standalone_mode)
