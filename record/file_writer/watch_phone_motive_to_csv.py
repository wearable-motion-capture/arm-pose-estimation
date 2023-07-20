import logging
import queue
from datetime import datetime
from pathlib import Path

import numpy as np

import config
from stream.listener.motive import MotiveListener
from stream.publisher.watch_phone import WatchPhonePublisher
from stream.publisher.motive_to_unity import MotiveToUnity
from utility import messaging


def watch_phone_motive_to_csv(sensor_q: queue,
                              motive_listener: MotiveListener,
                              dual_publisher: WatchPhonePublisher,
                              debug_motive_publisher: MotiveToUnity = None):
    tag = "REC WATCH PHONE MOTIVE"

    # create data header
    slp = messaging.WATCH_PHONE_IMU_LOOKUP
    header = ",".join(motive_listener.get_ground_truth_header() + list(slp.keys()))

    # create data filepath
    dirpath = Path(config.PATHS["cache"]) / "watch_phone_motive_rec"
    if not dirpath.exists():
        dirpath.mkdir(parents=True)
    filename = "watch_phone_motive_rec_{}.csv".format(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    # open the file
    with open(dirpath / filename, 'w') as csvfile:
        # first row are the column names
        csvfile.write(header + "\n")

        # now, wait for data
        start = datetime.now()
        count = 0
        logging.info("[{}] starting to write to {}".format(tag, filename))
        while 1:
            # get IMU data and more
            row = sensor_q.get()
            if row is None:
                continue

            # get ground truth
            gt_msg = motive_listener.get_ground_truth()
            if gt_msg is None:
                continue

            # visualize if debug publishers are available
            if dual_publisher is not None:
                msg = dual_publisher.row_to_arm_pose(row)
                dual_publisher.send_np_msg(msg)
            if debug_motive_publisher is not None:
                debug_motive_publisher.send_np_msg(gt_msg[:-4])  # skip the hip rotation

            # write everything to file
            s = ",".join(map(str, np.hstack([gt_msg, row]))) + "\n"

            csvfile.write(s)
            count += 1

            # second-wise updates to determine message frequency
            now = datetime.now()
            if (now - start).seconds >= 5:
                start = now
                logging.info("[{}] wrote {} lines".format(tag, count))
