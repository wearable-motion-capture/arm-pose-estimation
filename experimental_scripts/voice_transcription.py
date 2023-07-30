import argparse
import logging
import queue

import wear_mocap_ape.config as config
from wear_mocap_ape.stream.listener.audio import AudioListener
from wear_mocap_ape.stream.publisher.audio_udp import AudioUDP

# enable logging
logging.basicConfig(level=logging.INFO)
# parse command line arguments
parser = argparse.ArgumentParser(description='streams microphone data from the watch in standalone mode, '
                                             'transcribes it, and checks for a list of keywords for voice commands.')
# Required IP argument
parser.add_argument('ip', type=str, help=f'put your local IP here.')
args = parser.parse_args()
ip = args.ip

keyword_q = queue.Queue()

wp_audio = AudioListener(
    ip=ip,
    port=config.PORT_LISTEN_WATCH_AUDIO
)
wp_audio.transcription_loop(keyword_q)

pub_audio = AudioUDP(ip=ip)
pub_audio.stream_loop(keyword_q)
