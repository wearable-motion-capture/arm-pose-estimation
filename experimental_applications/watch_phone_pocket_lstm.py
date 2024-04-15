import argparse
import atexit
import logging
import queue
import signal
import threading

from wear_mocap_ape import config
from wear_mocap_ape.data_deploy.nn import deploy_models
from wear_mocap_ape.data_types import messaging
from wear_mocap_ape.stream.listener.imu import ImuListener
from wear_mocap_ape.stream.publisher.watch_phone_pocket_nn_udp import WatchPhonePocketNnUDP


def run_watch_phone_pocket_nn_udp(ip: str, smooth: int, stream_mc: bool) -> WatchPhonePocketNnUDP:
    # data for left-hand mode
    q = queue.Queue()

    # listen for imu data from phone and watch
    imu_listener = ImuListener(
        ip=ip,
        msg_size=messaging.watch_phone_imu_msg_len,
        port=config.PORT_LISTEN_WATCH_PHONE_IMU
    )
    imu_thread = threading.Thread(
        target=imu_listener.listen,
        args=(q,)
    )

    # process into arm pose and body orientation
    estimator = WatchPhonePocketNnUDP(ip=ip,
                                port=config.PORT_PUB_LEFT_ARM,
                                smooth=smooth,
                                model_hash=deploy_models.LSTM.WATCH_PHONE_POCKET.value,
                                stream_mc=stream_mc,
                                mc_samples=60)
    udp_thread = threading.Thread(
        target=estimator.processing_loop,
        args=(q,)
    )

    imu_thread.start()
    udp_thread.start()

    def terminate_all(*args):
        imu_listener.terminate()
        estimator.terminate()

    # make sure all handler exit on termination
    atexit.register(terminate_all)
    signal.signal(signal.SIGTERM, terminate_all)
    signal.signal(signal.SIGINT, terminate_all)

    return estimator


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Parse arguments
    parser = argparse.ArgumentParser(description='')

    # Required IP argument
    parser.add_argument('ip', type=str, help=f'put your local IP here.')
    parser.add_argument('smooth', nargs='?', type=int, default=5, help=f'smooth predicted trajectories')
    parser.add_argument('--stream_mc', action='store_true')
    parser.add_argument('--no-stream_mc', dest='stream_mc', action='store_false')
    parser.set_defaults(stream_mc=True)

    args = parser.parse_args()

    ip_arg = args.ip
    smooth_arg = args.smooth
    stream_mc_arg = args.stream_mc

    # run the predictions
    run_watch_phone_pocket_nn_udp(ip_arg, smooth_arg, stream_mc_arg)
