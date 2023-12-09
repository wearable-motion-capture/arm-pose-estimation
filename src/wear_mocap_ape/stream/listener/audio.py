import logging
import socket
import time

import pyaudio
import threading, queue

from datetime import datetime
from google.cloud import speech_v1 as speech

import wear_mocap_ape.config as config
import wear_mocap_ape.utility as utility
from wear_mocap_ape import data_types
from wear_mocap_ape.data_types import voice_commands


class AudioListener:
    def __init__(self, ip: str, port: int = config.PORT_LISTEN_AUDIO, tag: str = "AUDIO"):
        # Audio recording parameters
        # See https://github.com/googleapis/python-speech/blob/main/samples/microphone/transcribe_streaming_infinite.py
        self.__sample_rate = 44000
        self.__chunk_size = 2048  # int(16000 / 10)  # 100ms
        self.__language_code = "en-US"  # a BCP-47 language tag
        self.__ip = ip
        self.__port = port  # the dual Port
        self.__tag = tag
        self.__active = False

    def terminate(self):
        self.__active = False

    def play_stream_loop(self):
        """
        play back streamed mic data
        """
        # listener and audio player run in separate threads.
        # Listener fills the queue, audio player empties it
        q_in = queue.Queue()
        # the listener fills the que with transmitted smartwatch data
        t_mic_listen = threading.Thread(
            target=self.__stream_mic,
            args=(q_in,)
        )
        t_play = threading.Thread(
            target=self.__play_audio,
            args=(q_in,)
        )
        t_mic_listen.start()
        t_play.start()

    def transcription_loop(self, q_out: queue):
        """
        transcribes streamed mic data and logs keywords
        :param q_out: puts transcribed keywords into the queue
        """
        # listener and predictor run in separate threads. Listener fills the queue, predictor empties it
        q_in = queue.Queue()
        # the listener fills the que with transmitted smartwatch data
        t_mic_listen = threading.Thread(
            target=self.__stream_mic,
            args=(q_in,)
        )
        t_trans = threading.Thread(
            target=self.__transcribe,
            args=(q_in, q_out)
        )
        t_mic_listen.start()
        t_trans.start()

    def __stream_mic(self, q_out: queue):
        """
        :param q_out: listens to mic data sent from the smartwatch and fills the queue with whatever is received
        """

        # create a socket to listen on
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.bind((self.__ip, self.__port))
        s.settimeout(5)

        # if stream stopped listen again to wait for a new stream
        logging.info(f"[{self.__tag}] listening on {self.__ip} - {self.__port}")

        # begin receiving the data
        start = datetime.now()
        dat = 0
        self.__active = True
        while self.__active:
            # second-wise updates to determine message frequency
            now = datetime.now()
            if (now - start).seconds >= 5:
                start = now
                logging.info(f"[{self.__tag}] stream mic {dat / 5} Hz")
                dat = 0

            # this function waits
            try:
                data, _ = s.recvfrom(self.__chunk_size)
            except socket.timeout:
                logging.info(f"[{self.__tag}] timed out")
                continue

            if len(data) < self.__chunk_size:
                break

            if data:
                q_out.put(data)
                dat += 1
            else:
                break

        logging.info(f"[{self.__tag}] stream mic closed")

    def __play_audio(self, q: queue):
        # Instantiate PyAudio and initialize PortAudio system resources
        p = pyaudio.PyAudio()
        # Open output audio stream
        stream = p.open(format=pyaudio.paALSA,
                        frames_per_buffer=self.__chunk_size,
                        channels=1,
                        rate=self.__sample_rate,
                        output=True)
        logging.info(f"[{self.__tag}] audio player waiting for data")

        self.__active = True
        while self.__active:
            # get the next smartwatch data row from the queue
            row = q.get()
            if row:
                stream.write(row)
                time.sleep(0.01)
            else:
                break

        # Close the stream
        stream.close()
        # Release PortAudio system resources
        p.terminate()

    def __transcribe(self, q_in: queue, q_out: queue):
        client = speech.SpeechClient()

        aconfig = speech.RecognitionConfig({
            "encoding": speech.RecognitionConfig.AudioEncoding.LINEAR16,
            "sample_rate_hertz": self.__sample_rate,
            "language_code": self.__language_code
        })

        streaming_config = speech.StreamingRecognitionConfig({
            "config": aconfig,
            "interim_results": False
        })
        logging.info("[THREAD TRANSCRIBE] waiting for data")

        # infinite loop to transcribe everything that comes in
        while True:
            stream = []
            while not q_in.empty():
                stream.append(q_in.get())
            requests = (
                speech.StreamingRecognizeRequest({"audio_content": chunk}) for chunk in stream
            )

            responses = client.streaming_recognize(
                config=streaming_config,
                requests=requests
            )

            for response in responses:
                if not response.results:
                    break
                result = response.results[0]
                if not result.alternatives:
                    continue

                transcript = result.alternatives[0].transcript
                phrase = str(transcript).lower()
                logging.info("[THREAD TRANSCRIBE] {}".format(phrase))

                for k, v in voice_commands.KEY_PHRASES.items():
                    if k in phrase:
                        q_out.put(v)
                        logging.info("[THREAD TRANSCRIBE] sent {} queue size {}".format(k, q_out.qsize()))

            time.sleep(1.8)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    wp_audio = AudioListener(ip="192.168.1.162", port=config.PORT_LISTEN_AUDIO)
    wp_audio.play_stream_loop()
