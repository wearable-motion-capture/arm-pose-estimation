import argparse
import atexit
import logging
import queue
import signal
import threading

import wear_mocap_ape.config as config
from wear_mocap_ape.estimate.watch_phone_uarm_nn import WatchPhoneUarmNN
from wear_mocap_ape.stream.listener.imu import ImuListener
from wear_mocap_ape.stream.publisher.imu_udp import IMUPublisherUDP
from wear_mocap_ape.stream.publisher.watch_phone_uarm_nn_udp import WatchPhoneUarmNnUDP
from wear_mocap_ape.data_types import messaging


def run_watch_phone_uarm_nn_udp(ip, smooth, stream_mc):
    # the listener fills the que with received and parsed smartwatch data
    lstn = ImuListener(
        ip=ip,
        msg_size=messaging.watch_phone_imu_msg_len,
        port=config.PORT_LISTEN_WATCH_PHONE_IMU
    )
    sensor_q = lstn.listen_in_thread()

    # the estimator transforms sensor data into pose estimates and fills them into the output queue
    est = WatchPhoneUarmNN(
        smooth=smooth,
        stream_mc=stream_mc,
    )
    msg_q = est.process_in_threat(sensor_q)

    # the publisher publishes pose estimates from the queue via UDP
    pub = IMUPublisherUDP(
        ip=ip,
        port=config.PORT_PUB_LEFT_ARM,
    )
    pub.publish_in_thread(msg_q)

    # wait for any key to end the threads
    input("press enter to exit")
    lstn.terminate()
    est.terminate()
    pub.terminate()


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
    parser.add_argument('smooth', nargs='?', type=int, default=10, help=f'smooth predicted trajectories')
    args = parser.parse_args()

    run_watch_phone_uarm_nn_udp(ip=args.ip, smooth=args.smooth, stream_mc=True)
