import logging
import queue
from datetime import datetime
from pathlib import Path

import wear_mocap_ape.config as config
from wear_mocap_ape.utility import messaging


def watch_to_csv(sensor_q: queue):
    tag = "REC WATCH"
    slp = messaging.WATCH_ONLY_IMU_LOOKUP
    header = ",".join(slp.keys())

    dirpath = Path(config.paths["cache"]) / "watch_rec"

    if not dirpath.exists():
        dirpath.mkdir()

    filename = "watch_rec_{}.csv".format(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    logging.info("[{}] starting to write to {}".format(tag, filename))
    start = datetime.now()
    count = 0
    with open(dirpath / filename, 'w') as csvfile:
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
                logging.info("[{}] wrote {} lines".format(tag, count))
