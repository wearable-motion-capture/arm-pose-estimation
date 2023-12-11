import argparse
import atexit
import logging
import queue
import signal
import threading

from wear_mocap_ape import config
from wear_mocap_ape.data_types import messaging
from wear_mocap_ape.stream.listener.imu import ImuListener
from wear_mocap_ape.stream.publisher.watch_phone_pocket_kalman_udp import WatchPhonePocketKalmanUDP


def run_watch_phone_pocket_kalman(ip: str, smooth: int, stream_mc: bool) -> WatchPhonePocketKalmanUDP:
    # data for left-hand mode
    left_q = queue.Queue()

    # listen for imu data from phone and watch
    imu_l = ImuListener(
        ip=ip,
        msg_size=messaging.watch_phone_imu_msg_len,
        port=config.PORT_LISTEN_WATCH_PHONE_IMU_LEFT
    )
    l_thread = threading.Thread(
        target=imu_l.listen,
        args=(left_q,)
    )

    # process into arm pose and body orientation
    kpp = WatchPhonePocketKalmanUDP(ip=ip,
                                    smooth=smooth,
                                    num_ensemble=48,
                                    port=config.PORT_PUB_LEFT_ARM,
                                    window_size=10,
                                    stream_mc=stream_mc,
                                    model_hash="SW-model-sept-4")
    p_thread = threading.Thread(
        target=kpp.processing_loop,
        args=(left_q,)
    )

    l_thread.start()
    p_thread.start()

    def terminate_all(*args):
        imu_l.terminate()
        kpp.terminate()

    # make sure all handler exit on termination
    atexit.register(terminate_all)
    signal.signal(signal.SIGTERM, terminate_all)
    signal.signal(signal.SIGINT, terminate_all)
    return kpp


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Instantiate the parser
    parser = argparse.ArgumentParser(description='')

    # Required IP argument
    parser.add_argument('ip', type=str, help=f'put your local IP here.')
    parser.add_argument('smooth', nargs='?', type=int, default=5, help=f'smooth predicted trajectories')
    parser.add_argument('stream_mc', nargs='?', type=bool, default=True,
                        help=f'whether you want to stream the full pose ensemble')
    args = parser.parse_args()

    run_watch_phone_pocket_kalman(ip=args.ip, smooth=args.smooth, stream_mc=args.stream_mc)
