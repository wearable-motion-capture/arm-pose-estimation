# arm-pose-estimator

This repository provides code to run computations that estimate the arm pose. It takes input from the smartphone and smartwatch applications, processes the raw data stream, and outputs the visualization on Unity. 

Associated repositories:
* [sensor-stream-apps](https://github.com/wearable-motion-capture/sensor-stream-apps) provides the apps to stream sensor readings from wearable devices to a remote machine.
* [arm-pose-visualization](https://github.com/wearable-motion-capture/arm-pose-visualization) offers a real-time visualization using Unity.

## Install and Use

To utilize the "Dual" mode (smartwatch + smartphone), run the [`stream_dual.py`](https://github.com/wearable-motion-capture/arm-pose-estimation/blob/main/stream_dual.py) script. 

For "Standalone" mode (smartwatch), run the [`stream_standalone.py`](https://github.com/wearable-motion-capture/arm-pose-estimation/blob/main/stream_standalone.py) script. 

Please read
the [step-by-step instructions](https://docs.google.com/document/d/1ayMBF9kDCB9rlcrqR0sPumJhIVJgOF-SENTdoE4a6DI/edit?usp=sharing) for a detailed instruction guide.
