import argparse
import atexit
import logging
import queue
import signal
import threading

import wear_mocap_ape.config as config
from wear_mocap_ape.stream.listener.imu import ImuListener
from wear_mocap_ape.data_types import messaging
from wear_mocap_ape.stream.publisher.watch_phone_uarm_udp import WatchPhoneUarmUDP


def run_watch_phone_uarm_udp(ip, smooth):
    # data processing happens in independent threads.
    # We exchange data via queues.
    left_q = queue.Queue()  # data for left-hand mode

    # left listener
    imu_l = ImuListener(
        ip=ip,
        msg_size=messaging.watch_phone_imu_msg_len,
        port=config.PORT_LISTEN_WATCH_PHONE_IMU,
    )
    l_thread = threading.Thread(
        target=imu_l.listen,
        args=(left_q,)
    )

    # left publisher
    wp2ul = WatchPhoneUarmUDP(
        ip=ip,
        port=config.PORT_PUB_LEFT_ARM,
        smooth=smooth
    )
    ul_thread = threading.Thread(
        target=wp2ul.processing_loop,
        args=(left_q,)
    )
    l_thread.start()
    ul_thread.start()

    def terminate_all(*args):
        imu_l.terminate()
        wp2ul.terminate()

    # make sure all handler exit on termination
    atexit.register(terminate_all)
    signal.signal(signal.SIGTERM, terminate_all)
    signal.signal(signal.SIGINT, terminate_all)

    return wp2ul


if __name__ == "__main__":
    # enable basic logging
    logging.basicConfig(level=logging.INFO)

    # Instantiate the parser
    parser = argparse.ArgumentParser(description='streams data from the watch in standalone mode')

    # Required IP argument
    parser.add_argument('ip', type=str,
                        help=f'put your local IP here. '
                             f'The script will publish arm '
                             f'pose data on PORT {config.PORT_PUB_LEFT_ARM} (left hand)')
    parser.add_argument('smooth', nargs='?', type=int, default=5, help=f'smooth predicted trajectories')
    args = parser.parse_args()

    run_watch_phone_uarm_udp(ip=args.ip, smooth=args.smooth)
