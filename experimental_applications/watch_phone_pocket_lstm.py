import argparse
import logging

from wear_mocap_ape import config
from wear_mocap_ape.data_deploy.nn import deploy_models
from wear_mocap_ape.data_types import messaging
from wear_mocap_ape.stream.listener.imu import ImuListener
from wear_mocap_ape.estimate.watch_phone_pocket_nn import WatchPhonePocketNN
from wear_mocap_ape.stream.publisher.pose_est_udp import PoseEstPublisherUDP


def run_watch_phone_pocket_nn_udp(ip: str, smooth: int) -> WatchPhonePocketNN:
    # listen for imu data from phone and watch
    imu_listener = ImuListener(
        ip=ip,
        msg_size=messaging.watch_phone_imu_msg_len,
        port=config.PORT_LISTEN_WATCH_PHONE_IMU
    )
    sensor_q = imu_listener.listen_in_thread()

    # process into arm pose and body orientation
    estimator = WatchPhonePocketNN(smooth=smooth,
                                   model_hash=deploy_models.LSTM.WATCH_PHONE_POCKET.value,
                                   add_mc_samples=True,
                                   monte_carlo_samples=60)
    msg_q = estimator.process_in_thread(sensor_q)

    # the publisher publishes pose estimates from the queue via UDP
    pub = PoseEstPublisherUDP(
        ip=ip,
        port=config.PORT_PUB_LEFT_ARM
    )
    pub.publish_in_thread(msg_q)

    # wait for any key to end the threads
    input("[TERMINATION TRIGGER] press enter to exit")
    imu_listener.terminate()
    estimator.terminate()
    pub.terminate()

    return estimator


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Parse arguments
    parser = argparse.ArgumentParser(description='')

    # Required IP argument
    parser.add_argument('ip', type=str, help=f'put your local IP here.')
    parser.add_argument('smooth', nargs='?', type=int, default=5, help=f'smooth predicted trajectories')
    parser.set_defaults(stream_mc=True)

    args = parser.parse_args()

    ip_arg = args.ip
    smooth_arg = args.smooth

    # run the predictions
    run_watch_phone_pocket_nn_udp(ip_arg, smooth_arg)
