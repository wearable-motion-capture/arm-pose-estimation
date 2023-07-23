# arm-pose-estimator

This repository provides code to run computations that estimate the arm pose. It takes input from the smartphone and
smartwatch applications, processes the raw data stream, and outputs the visualization on Unity.

Associated repositories:

* [sensor-stream-apps](https://github.com/wearable-motion-capture/sensor-stream-apps) provides the apps to stream sensor
  readings from wearable devices to a remote machine.
* [arm-pose-visualization](https://github.com/wearable-motion-capture/arm-pose-visualization) offers a real-time
  visualization using Unity.

## Install and Use

To utilize the "Dual" mode (smartwatch + smartphone), run
the [stream_dual.py](https://github.com/wearable-motion-capture/arm-pose-estimation/blob/main/stream_dual.py) script.
For "Standalone" mode (smartwatch), run
the [stream_standalone.py](https://github.com/wearable-motion-capture/arm-pose-estimation/blob/main/stream_standalone.py)
script.

Please read
the [step-by-step instructions](https://docs.google.com/document/d/1ayMBF9kDCB9rlcrqR0sPumJhIVJgOF-SENTdoE4a6DI/edit?usp=sharing)
for a detailed instruction guide.

## Experimental Modes

We augment arm pose estimations with further modes to interface with ROS or recognize voice commands. These are marked as experimental because they depend on external APIs, such as ROS and the Google Cloud transcription service.

### Audio Transcription

This project can receive the microphone data from our smartwatch app and transcribe it for voice commands etc.
For this, we use the Google Cloud transcription service.

If you don't have an account, follow
the [quick start guide](https://cloud.google.com/speech-to-text/docs/before-you-begin?hl=en#setting_up_your_google_cloud_platform_project)

Then, add `GOOGLE_APPLICATION_CREDENTIALS=/path/to/google_auth.json` to your environment variables before running the script.

Run the transcription service with [stream_transcription_commands.py](https://github.com/wearable-motion-capture/arm-pose-estimation/blob/main/experimental_modes/stream_transcription_commands.py). It will publish IDs of recognized keywords (voice commands) via UDP on the port 50006.

Set voice command keyword IDs in [voice_commands.py](https://github.com/wearable-motion-capture/arm-pose-estimation/blob/main/utility/voice_commands.py).