import logging

import wear_mocap_ape.config as config
from wear_mocap_ape.estimate.watch_only import WatchOnlyNN
from wear_mocap_ape.stream.publisher.imu_udp import IMUPublisherUDP

from wear_mocap_ape.stream.listener.imu import ImuListener
from wear_mocap_ape.data_types import messaging

import argparse


def run_watch_only_nn_udp(ip, smooth, stream_mc):
    # the listener fills the que with received and parsed smartwatch data
    lstn = ImuListener(
        ip=ip,
        msg_size=messaging.watch_only_imu_msg_len,
        port=config.PORT_LISTEN_WATCH_IMU  # the smartwatch app sends on this PORT
    )
    sensor_q = lstn.listen_in_thread()

    # the estimator transforms sensor data into pose estimates and fills them into the output queue
    est = WatchOnlyNN(
        smooth=smooth,
        add_mc_samples=stream_mc
    )
    msg_q = est.process_in_thread(sensor_q)

    # the publisher publishes pose estimates from the queue via UDP
    pub = IMUPublisherUDP(
        ip=ip,
        port=config.PORT_PUB_LEFT_ARM
    )
    pub.publish_in_thread(msg_q)

    # wait for any key to end the threads
    input("[TERMINATION TRIGGER] press enter to exit")
    lstn.terminate()
    est.terminate()
    pub.terminate()


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

    run_watch_only_nn_udp(ip=args.ip, smooth=10, stream_mc=True)
