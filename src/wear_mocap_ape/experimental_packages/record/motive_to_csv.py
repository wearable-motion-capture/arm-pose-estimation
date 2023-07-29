import logging
import threading
import time
from datetime import datetime
from pathlib import Path
import wear_mocap_ape.config as config
from wear_mocap_ape.stream.listener.motive_q import MotiveQListener
from wear_mocap_ape.stream.publisher.motive_to_unity import MotiveToUnity


def motive_to_csv(motive_listener: MotiveQListener, debug_motive_publisher: MotiveToUnity = None):
    tag = "REC MOTIVE"

    # create data header
    header = ",".join(motive_listener.get_extended_ground_truth_header())

    # create data filepath
    dirpath = Path(config.paths["cache"]) / "motive_rec"
    if not dirpath.exists():
        dirpath.mkdir(parents=True)
    filename = "motive_rec_{}.csv".format(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    # open the file
    with open(dirpath / filename, 'w') as csvfile:
        # first row are the column names
        csvfile.write(header + "\n")

        # now, wait for data
        start = datetime.now()
        count = 0
        logging.info("[{}] starting to write to {}".format(tag, filename))
        while 1:
            # get ground truth
            gt_msg = motive_listener.get_ground_truth()
            ext_gt_msg = motive_listener.get_extended_ground_truth()
            if gt_msg is None or ext_gt_msg is None:
                continue

            # visualize if debug publishers are available
            if debug_motive_publisher is not None:
                debug_motive_publisher.send_np_msg(gt_msg)

            # write everything to file
            s = ",".join(map(str, ext_gt_msg)) + "\n"
            csvfile.write(s)
            count += 1

            # second-wise updates to determine message frequency
            now = datetime.now()
            if (now - start).seconds >= 5:
                start = now
                logging.info("[{}] wrote {} lines".format(tag, count))
            time.sleep(0.01)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ml = MotiveListener()
    ml_thread = threading.Thread(target=ml.stream_loop)
    ml_thread.start()
    mp = MotiveToUnity()
    motive_to_csv(ml, mp)
