import logging
import queue
import threading

from record.file_writer.watch_to_csv import watch_to_csv
from stream.listener.watch_imu import listen_for_watch_imu

if __name__ == "__main__":
    # start ros node
    logging.basicConfig(level=logging.INFO)

    # listener and predictor run in separate threads. Listener fills the queue, predictor empties it
    sensor_que = queue.Queue()

    # the listener fills the que with transmitted smartwatch data
    sensor_listener = threading.Thread(
        target=listen_for_watch_imu,
        args=(sensor_que,)
    )
    sensor_listener.start()

    watch_to_csv(sensor_q=sensor_que)
