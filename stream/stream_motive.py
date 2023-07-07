import logging
import threading

from stream.listener.motive import MotiveListener
from stream.publisher.motive_gt import publish_gt

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    motive_listener = MotiveListener()
    gt_publisher = threading.Thread(
        target=publish_gt,
        args=(motive_listener,)
    )
    gt_publisher.start()

    motive_listener.start()
