import logging
import queue

from stream.listener.audio import AudioListener
from stream.publisher.audio import AudioPublisher

logging.basicConfig(level=logging.INFO)
keyword_q = queue.Queue()

wp_audio = AudioListener()
wp_audio.transcription_loop(keyword_q)

pub_audio = AudioPublisher()
pub_audio.stream_loop(keyword_q)
