import logging
import os
import threading
from datetime import datetime
import queue
from pynput.keyboard import Listener
import config
from utility import messaging

PORT = 50003
TAG = "REC ALL"

class imu_ppg_recorder:

    def __init__(self, sensor_q: queue):
        self.__sensor_q = sensor_q
        self.__hand = "open"
        self.__recording = False
        self.__active = True


    def on_release(self, key):
        print('{0} release'.format(key))
        if hasattr(key, "char"):
            if key.char == 'o':
                self.__hand = "open"
                logging.info("[{}] hand open".format(TAG))
            elif key.char == 'c':
                self.__hand = "closed"
                logging.info("[{}] hand closed".format(TAG))
            elif key.char == 'r':
                self.__recording = not self.__recording
                if self.__recording:
                    logging.info("[{}] start recording messages".format(TAG))
                else:
                    logging.info("[{}] stop recording sw messages".format(TAG))
            elif key.char == 'a':
                self.__active = False

    def listen_to_keyboard(self):
        # Collect events until released
        with Listener(on_release=self.on_release) as listener:
            logging.info("[{}] listening to keyboard".format(TAG))
            listener.join()

    def start(self):
        update_thread = threading.Thread(target=self.update_routine)
        update_thread.start()
        keyboard_thread = threading.Thread(target=self.listen_to_keyboard)
        keyboard_thread.start()

    def update_routine(self):

        start = datetime.now()
        slp = messaging.sw_standalone_imu_lookup
        dat = 0

        # name of the dataset
        header = "@relation SW_PPG." + start.strftime("%Y-%m-%d_%H-%M-%S") + "\n\n"
        # attributes are column names
        for k, v in slp.items():
            header += "@attribute " + k + " real\n"
        header += "@attribute hand {open, closed}\n\n"  # add hand parameter

        dirpath = os.path.join(config.paths["cache"], "PPG_rec")
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        filepath = os.path.join(dirpath, start.strftime("%Y-%m-%d_%H-%M-%S") + ".arff")

        csvfile = open(filepath, 'w')
        csvfile.write(header)

        # this loops while the socket is listening and/or receiving data
        while self.__active:

            # get the most recent smartwatch data row from the queue
            row = None
            while not self.__sensor_q.empty():
                row = self.__sensor_q.get()

            # process received data
            if row is not None:

                # second-wise updates to determine message frequency
                now = datetime.now()
                if (now - start).seconds >= 5:
                    start = now
                    logging.info("[{}] {} Hz".format(TAG, dat / 5))
                    dat = 0

                if self.__recording:
                    s = ",".join(map(str, row))
                    s += "," + self.__hand + "\n"
                    csvfile.write(s)

        csvfile.close()
        logging.info("[{}] closed file - done".format(TAG))
