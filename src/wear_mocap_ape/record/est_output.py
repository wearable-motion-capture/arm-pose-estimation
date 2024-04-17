import logging
import queue
import datetime
import threading

from pathlib import Path


class EstOutputRecorder:
    def __init__(self, file: Path, tag: str = "REC EST OUTPUT"):
        self.__file = Path(file)
        self.__tag = tag
        self._active = False

        # the header matches the msg in process_msg
        header = [
            "time",
            "hand_quat_w", "hand_quat_x", "hand_quat_y", "hand_quat_z",
            "hand_orig_rh_x", "hand_orig_rh_y", "hand_orig_rh_z",
            "larm_quat_rh_w", "larm_quat_rh_x", "larm_quat_rh_y", "larm_quat_rh_z",
            "larm_orig_rh_x", "larm_orig_rh_y", "larm_orig_rh_z",
            "uarm_quat_rh_w", "uarm_quat_rh_x", "uarm_quat_rh_y", "uarm_quat_rh_z",
            "uarm_orig_rh_x", "uarm_orig_rh_y", "uarm_orig_rh_z",
            "hips_quat_g_w", "hips_quat_g_x", "hips_quat_g_y", "hips_quat_g_z"
        ]

        if not self.__file.parent.exists():
            raise UserWarning(f"Directory does not exist {file.parent}")

        with open(self.__file, 'w') as fd:
            fd.write(",".join(header) + "\n")

        logging.info(f"[{self.__tag}] Writing to file {self.__file}")

    def terminate(self):
        self._active = False

    def record_in_thread(self, msg_q: queue):
        t = threading.Thread(target=self.write_queue_to_csv, args=(msg_q,))
        t.start()

    def write_queue_to_csv(self, msg_q: queue):
        self._active = True
        while self._active:
            try:
                # get most recent pose estimate from queue
                msg = msg_q.get(timeout=2)
            except queue.Empty:
                logging.info(f"[{self.__tag}] no data")
                continue

            # write data to CSV
            msg = msg.tolist()
            with open(self.__file, 'a') as fd:
                msg.insert(0, datetime.datetime.now())
                msg = [str(x) for x in msg]
                fd.write(",".join(msg) + "\n")
