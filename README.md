# tello-rl-tracker
Autonomous deep reinforcement learning agent tracker for Tello drones. It uses python 3 and [TelloPy](https://github.com/hanyazou/TelloPy).
The starting point is [Ubotica telloCV](https://github.com/Ubotica/telloCV), Nov 6 2018.

It uses a neural network to identify a face in the image and infer its center and radius, then (STILL TO BE IMPLEMENTED) uses a deepRL agent to drive the tello in such a way to keep the face at the center of the image.

## Installation
WORK IN PROGESS!

TelloPy:
```
git clone https://github.com/hanyazou/TelloPy
cd TelloPy
python setup.py bdist_wheel
pip install dist/tellopy-*.dev*.whl --upgrade
```

## Control commands
All control commands are described in telloCV.py.

## Files
telloCV.py: controller

cnn_tracker.py: NN tracker, at the moment it tracks the most prominent person in the image.

tracker.py: tracker of [Ubotica telloCV](https://github.com/Ubotica/telloCV), Nov 6 2018. (In order to work requires a few changes in telloCV.py)

eager_few_shot_od_training_tf2_colab.ipynb: Colab notebook taken from [tensorflow repo](https://github.com/tensorflow/models/tree/master/research/object_detection/colab_tutorials), used to perform transfer learning.

