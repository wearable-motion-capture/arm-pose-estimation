import atexit
import logging
import queue
import signal
import threading
import wear_mocap_ape.config as config

from wear_mocap_ape.stream.publisher.watch_only_udp import WatchOnlyNnUDP
from wear_mocap_ape.stream.listener.imu import ImuListener
from wear_mocap_ape.data_types import messaging

import argparse


def run_watch_phone_watch_only_nn_udp(ip, smooth, stream_mc):
    # listener and publisher run in separate threads. Listener fills the queue, publisher empties it
    q = queue.Queue()

    # listen for imu data from phone and watch
    imu_l = ImuListener(
        ip=ip,
        msg_size=messaging.watch_phone_imu_msg_len,
        port=config.PORT_LISTEN_WATCH_PHONE_IMU
    )
    imu_thread = threading.Thread(
        target=imu_l.listen,
        args=(q,)
    )

    # the publisher estimates arm poses from queued data and publishes them via UDP to given IP and PORT
    estimator = WatchOnlyNnUDP(
        ip=ip,
        port=config.PORT_PUB_LEFT_ARM,
        smooth=smooth,
        stream_mc=stream_mc,
        mc_samples=50,
        watch_phone=True
    )
    udp_thread = threading.Thread(
        target=estimator.processing_loop,
        args=(q,)
    )

    imu_thread.start()
    udp_thread.start()

    def terminate_all(*args):
        imu_l.terminate()
        estimator.terminate()

    # make sure all handler exit on termination
    atexit.register(terminate_all)
    signal.signal(signal.SIGTERM, terminate_all)
    signal.signal(signal.SIGINT, terminate_all)
    return estimator


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Instantiate the parser
    parser = argparse.ArgumentParser(description='streams data from the watch in standalone mode')

    # Required IP argument
    parser.add_argument('ip', type=str,
                        help=f'put your local IP here. '
                             f'The script will publish arm '
                             f'pose data on PORT {config.PORT_PUB_LEFT_ARM}')
    args = parser.parse_args()

    run_watch_phone_watch_only_nn_udp(ip=args.ip, smooth=5, stream_mc=True)
