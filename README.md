# Arm Pose Estimator Module

This repository is part of the [wearable motion capture](https://github.com/wearable-motion-capture) framework and 
provides code to estimate the arm pose from smartwatch and smartphone sensor data. It opens network sockets to stream 
sensor data from our smartwatch and phone apps provided in [sensor-stream-apps](https://github.com/wearable-motion-capture/sensor-stream-apps).
You can then either record the arm poses into a `csv` file, or stream them to our [arm-pose-visualization](https://github.com/wearable-motion-capture/arm-pose-visualization) 
project for a real-time visualization.

Associated repositories:

* [sensor-stream-apps](https://github.com/wearable-motion-capture/sensor-stream-apps) provides the apps to stream sensor
  readings from wearable devices to a remote machine.
* [arm-pose-visualization](https://github.com/wearable-motion-capture/arm-pose-visualization) contains the real-time
  visualization of estimated arm poses using Unity.

## Install

Clone the git repository and install it with

```
pip install -e \path\to\project\root
```

## Use

Please see the scripts in
the [example_scripts](https://github.com/wearable-motion-capture/arm-pose-estimation/blob/main/example_scripts)
directory.

Currently, there are four scripts and two types of scripts.

The stream scripts require your local IP as input. Run them as, for example,

```
python3 stream_watch_only.py 192.168.1.123
```

The record scripts also require a file path. Recorded data will be stored into that file. Run them as:

```
python3 record_watch_only.py 192.168.1.123 path/to/file.csv
```

Namely, the scripts are:

* [stream_watch_only.py](https://github.com/wearable-motion-capture/arm-pose-estimation/blob/main/example_scripts/stream_watch_only.py)
  estimates arm poses from watch only data and publishes them via a UDP socket on given IP and PORT 46000.
* [record_watch_only](https://github.com/wearable-motion-capture/arm-pose-estimation/blob/main/example_scripts/record_watch_only.py)
  estimates arm poses from watch data only and writes them into a `.csv` file. Pass the filepath as a parameter.
* [stream_watch_phone.py](https://github.com/wearable-motion-capture/arm-pose-estimation/blob/main/example_scripts/stream_watch_phone.py)
  estimates arm poses from watch and phone data together and publishes them via a UDP socket on given IP and PORT 65000.
* [record_watch_phone.py](https://github.com/wearable-motion-capture/arm-pose-estimation/blob/main/example_scripts/record_watch_phone.py)
  estimates arm poses from watch and phone and writes them into a `.csv` file. Pass the filepath as a parameter.

In case you require more detailed instructions, please read
the [step-by-step guide](https://docs.google.com/document/d/1ayMBF9kDCB9rlcrqR0sPumJhIVJgOF-SENTdoE4a6DI/edit?usp=sharing).

## Experimental Scripts

We augment arm pose estimations with further modes. For example, to interface with ROS or to recognize voice commands.
These are marked as experimental because they are less extensively documented, and it is guaranteed that they will be
subject to future development. Further, they depend on external APIs, such as ROS and the Google Cloud transcription
service.
You can find the experimental scripts in the
[experimental_scripts](https://github.com/wearable-motion-capture/arm-pose-estimation/blob/main/experimental_scripts)
directory.

### Audio Transcription

This project can receive the microphone data from our smartwatch app and transcribe it for voice commands etc.
Run the transcription service
with [stream_transcription_commands.py](https://github.com/wearable-motion-capture/arm-pose-estimation/blob/main/experimental_scripts/stream_transcription_commands.py).
It will publish IDs of recognized keywords (voice commands) via UDP on the port 50006.

We use the Google Cloud transcription service. If you don't have an account, follow
the [quick start guide](https://cloud.google.com/speech-to-text/docs/before-you-begin?hl=en#setting_up_your_google_cloud_platform_project)

Then, add `GOOGLE_APPLICATION_CREDENTIALS=/path/to/google_auth.json` to your environment variables before running the
script.

The additional dependencies are for audio services are

```
pyaudio
google-cloud-speech
```

Install them via `pip`

Set voice command keyword IDs
in [voice_commands.py](https://github.com/wearable-motion-capture/arm-pose-estimation/blob/main/src/wear_mocap_ape/utility/voice_commands.py).

### ROS

This script requires `rospy`. For an example on how to publish data as a ROS node, see
[stream_ros_exp.py](https://github.com/wearable-motion-capture/arm-pose-estimation/blob/main/experimental_scripts/stream_ros_exp.py).
