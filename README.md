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

face_rec_tracker.py: uses a SVM and python's face_recognition to recognize faces, the binary SVM can be computed with "svm.py"

svm.py: Creates a SVM capable of recognize faces, at the top of the file is shown how one should organize the images. [Face recognition](https://github.com/ageitgey/face_recognition)

yolo_tracker.py: NN tracker, at the moment I'm trying to train a network to recognize faces (Work in progress, pay attention to distance measure in telloCV.py).

tracker.py: tracker of [Ubotica telloCV](https://github.com/Ubotica/telloCV), Nov 6 2018. (In order to work requires a few changes in telloCV.py)

yolo5_boxes.ipynb: Colab notebook taken from [Object detection](https://laptrinhx.com/how-to-train-a-custom-object-detection-model-with-yolo-v5-4206857070/), used to train object detector.

