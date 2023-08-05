import logging
import queue
from datetime import datetime
from pathlib import Path

from wear_mocap_ape.data_types import messaging


def arm_pose_to_csv(sensor_q: queue, dir_path: Path):
    tag = "REC WATCH"
    slp = messaging.WATCH_ONLY_IMU_LOOKUP
    header = ",".join(slp.keys())

    filename = "arm_pose_rec_{}.csv".format(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    logging.info(f"[{tag}] starting to write to {filename}")
    start = datetime.now()
    count = 0
    with open(dir_path / filename, 'w') as csvfile:
        csvfile.write(header + "\n")
        while 1:
            row = sensor_q.get()
            if row is not None:
                s = ",".join(map(str, row)) + "\n"
                csvfile.write(s)
                count += 1

            # second-wise updates to determine message frequency
            now = datetime.now()
            if (now - start).seconds >= 5:
                start = now
                logging.info(f"[{tag}] wrote {count} lines")
