import argparse
import logging

from wear_mocap_ape import config
from wear_mocap_ape.data_types import messaging
from wear_mocap_ape.estimate.watch_phone_pocket_kalman import WatchPhonePocketKalman
from wear_mocap_ape.stream.listener.imu import ImuListener
from wear_mocap_ape.stream.publisher.pose_est_udp import PoseEstPublisherUDP


def run_watch_phone_pocket_kalman(ip: str, smooth: int, stream_mc: bool):
    # the listener fills the que with received and parsed smartwatch data
    lstn = ImuListener(
        ip=ip,
        msg_size=messaging.watch_phone_imu_msg_len,
        port=config.PORT_LISTEN_WATCH_PHONE_IMU
    )
    sensor_q = lstn.listen_in_thread()

    # the estimator transforms sensor data into pose estimates and fills them into the output queue
    est = WatchPhonePocketKalman(
        model_path=config.PATHS["deploy"] / "kalman" / "SW-v3.8-model-436400",
        smooth=smooth,
        num_ensemble=48,
        window_size=10,
        add_mc_samples=stream_mc,
    )
    msg_q = est.process_in_thread(sensor_q)

    # the publisher publishes pose estimates from the queue via UDP
    pub = PoseEstPublisherUDP(
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
    parser = argparse.ArgumentParser(description='')

    # Required IP argument
    parser.add_argument('ip', type=str, help=f'put your local IP here.')
    parser.add_argument('smooth', nargs='?', type=int, default=5, help=f'smooth predicted trajectories')
    args = parser.parse_args()

    run_watch_phone_pocket_kalman(ip=args.ip, smooth=args.smooth, stream_mc=True)
