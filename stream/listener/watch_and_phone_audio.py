import logging
import socket
import time

import pyaudio
import threading, queue

from datetime import datetime
from google.cloud import speech_v1 as speech

import config
import utility.voice_commands


class WatchAndPhoneAudio:
    def __init__(self):
        # Audio recording parameters
        # See https://github.com/googleapis/python-speech/blob/main/samples/microphone/transcribe_streaming_infinite.py
        self.__sample_rate = 16000
        self.__chunk_size = 800  # int(16000 / 10)  # 100ms
        self.__language_code = "en-US"  # a BCP-47 language tag
        self.__ip = config.IP
        self.__port = 65001  # the dual Port
        self.__tag = "WATCH PHONE AUDIO"

    def run_transcription(self, q_out: queue):
        """
        :param q_out: puts transcribed keywords into the queue
        """
        # listener and predictor run in separate threads. Listener fills the queue, predictor empties it
        q_in = queue.Queue()
        # the listener fills the que with transmitted smartwatch data
        t_mic_listen = threading.Thread(
            target=self.stream_mic,
            args=(q_in,)
        )
        t_trans = threading.Thread(
            target=self.transcribe,
            args=(q_in, q_out)
        )
        t_mic_listen.start()
        t_trans.start()

    def stream_mic(self, q_out: queue):
        """
        :param q_out: listens to mic data sent from the smartwatch and fills the queue with whatever is received
        """

        # create a socket to listen on
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.bind((self.__ip, self.__port))

        # if stream stopped listen again to wait for a new stream
        logging.info(f"[{self.__tag}] listening on {self.__ip} - {self.__port}")

        # begin receiving the data
        start = datetime.now()
        dat = 0
        while 1:
            # second-wise updates to determine message frequency
            now = datetime.now()
            if (now - start).seconds >= 5:
                start = now
                logging.info(f"[{self.__tag}] stream mic {dat / 5} Hz")
                dat = 0

            # this function waits
            data, _ = s.recvfrom(self.__chunk_size)

            if len(data) < self.__chunk_size:
                break

            if data:
                q_out.put(data)
                dat += 1
            else:
                break

        logging.info(f"[{self.__tag}] stream mic closed")

    def play_audio(self, q: queue):
        # Instantiate PyAudio and initialize PortAudio system resources
        p = pyaudio.PyAudio()
        # Open output audio stream
        stream = p.open(format=pyaudio.paInt16,
                        frames_per_buffer=self.__chunk_size,
                        channels=1,
                        rate=self.__sample_rate,
                        output=True)
        logging.info(f"[{self.__tag}] audio player waiting for data")

        while True:
            # get the next smartwatch data row from the queue
            row = q.get()
            if row:
                stream.write(row)
            else:
                break

        # Close the stream
        stream.close()
        # Release PortAudio system resources
        p.terminate()

    def transcribe(self, q_in: queue, q_out: queue):
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

                for k, v in utility.voice_commands.commands.items():
                    if k in phrase:
                        q_out.put(v)
                        logging.info("[THREAD TRANSCRIBE] sent {} queue size {}".format(k, q_out.qsize()))

            time.sleep(1.8)


if __name__ == "__main__":
    # start ros node
    logging.basicConfig(level=logging.INFO)

    # listener and predictor run in separate threads. Listener fills the queue, predictor empties it
    que_in = queue.Queue()
    que_out = queue.Queue()

    wp_audio = WatchAndPhoneAudio()

    # the listener fills the que with transmitted smartwatch data
    mic_listener = threading.Thread(
        target=wp_audio.stream_mic,
        args=(que_in,)
    )

    player = threading.Thread(
        target=wp_audio.play_audio,
        args=(que_in,)
    )
    player.start()

    # transcriber = threading.Thread(
    #     target=transcribe,
    #     args=(que_in,que_out)
    # )
    # transcriber.start()

    mic_listener.start()
