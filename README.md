# tello-rl-tracker
Autonomous deep reinforcement learning agent tracker for Tello drones. It uses python 3 and [TelloPy](https://github.com/hanyazou/TelloPy).
The starting point is [Ubotica telloCV](https://github.com/Ubotica/telloCV).

It uses a neural network to identify a face in the image and infer it center and radius, then (STILL TO BE IMPLEMENTED) uses a deepRL agent to drive the tello in such a way to keep the face at the center of the image.

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
