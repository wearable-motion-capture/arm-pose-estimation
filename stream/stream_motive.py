import logging
import threading

from stream.listener.motive import MotiveListener
from stream.publisher.motive_gt import MotiveToUnity

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    motive_listener = MotiveListener()
    listen_thread = threading.Thread(
        target=motive_listener.stream_loop,
    )
    listen_thread.start()

    motive_publisher = MotiveToUnity()
    publish_thread = threading.Thread(
        target=motive_publisher.stream_loop,
        args=(motive_listener,)
    )
    publish_thread.start()
