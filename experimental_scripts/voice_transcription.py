import logging
import queue

import wear_mocap_ape.config as config
from wear_mocap_ape.stream.listener.audio import AudioListener
from wear_mocap_ape.stream.publisher.audio import AudioPublisher

logging.basicConfig(level=logging.INFO)
keyword_q = queue.Queue()

wp_audio = AudioListener(port=config.PORT_LISTEN_WATCH_AUDIO)
wp_audio.transcription_loop(keyword_q)

pub_audio = AudioPublisher(ip=config.IP_OWN)
pub_audio.stream_loop(keyword_q)
