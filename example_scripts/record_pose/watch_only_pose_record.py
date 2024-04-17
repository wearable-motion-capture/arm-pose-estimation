import argparse
import logging
import wear_mocap_ape.config as config

from wear_mocap_ape.estimate.watch_only import WatchOnlyNN
from wear_mocap_ape.record.est_output import EstOutputRecorder
from wear_mocap_ape.stream.listener.imu import ImuListener
from wear_mocap_ape.data_types import messaging

# basic logging outputs
logging.basicConfig(level=logging.INFO)

# read input arguments
parser = argparse.ArgumentParser(description='records data from the watch in standalone mode')
parser.add_argument('ip', type=str,
                    help=f'put your local IP here. '
                         f'The script will listen to watch data on '
                         f'PORT {config.PORT_LISTEN_WATCH_IMU}')
parser.add_argument('file', type=str,
                    help=f'recorded data will be written into this file')
args = parser.parse_args()

# the listener fills the queue with received and parsed smartwatch data
lstn = ImuListener(
    ip=args.ip,
    msg_size=messaging.watch_only_imu_msg_len,
    port=config.PORT_LISTEN_WATCH_IMU  # the smartwatch app sends on this PORT
)
sensor_q = lstn.listen_in_thread()

# the estimator transforms sensor data into pose estimates and fills them into the output queue
est = WatchOnlyNN(smooth=1, add_mc_samples=False)
msg_q = est.process_in_thread(sensor_q)

# the recorder writes the pose estimates to the given file path
rec = EstOutputRecorder(file=args.file)
rec.record_in_thread(msg_q=msg_q)

# wait for any key to end the threads
input("[TERMINATION TRIGGER] press enter to exit")
lstn.terminate()
est.terminate()
rec.terminate()
