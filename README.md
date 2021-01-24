# tello-ai-tracker
Autonomous tracker for Tello drones. It uses python 3.7 and [TelloPy](https://github.com/hanyazou/TelloPy).
The starting point is [Ubotica telloCV](https://github.com/Ubotica/telloCV), Nov 6 2018.

Identifies a known face in the image and infers its smallest enclosing rectangles, then (STILL TO BE IMPLEMENTED) uses a NN to drive the tello in such a way to keep the face at the center of the image.

## Installation
Install anaconda and then:
```
conda create -n <env_name> python=3.7
pip install -r requirements.txt
```

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

### Warning
The binarized svm in the repo, "svm_fam.bin", should be replaced with an svm fitted with your images, using the script "svm.py".
