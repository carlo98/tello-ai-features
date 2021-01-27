# tello-ai-tracker
Autonomous tracker for Tello drones. It works on python 3.7 and 3.6 and uses [TelloPy](https://github.com/hanyazou/TelloPy).

The starting point is [Ubotica telloCV](https://github.com/Ubotica/telloCV), Nov 6 2018.

Identifies a known face in the image, infers its smallest enclosing rectangle and keeps the face at the center of the image.

Implements frontal collision avoidance with NN, look at corresponding paragraph.

## Installation
Install anaconda and opencv and then:
```
conda create --name <env> python=3.7 # or python=3.6
conda activate <env>
conda install pip
pip install numpy av pynput face-recognition sklearn torch
```
TelloPy:
```
git clone https://github.com/hanyazou/TelloPy
cd TelloPy
python setup.py bdist_wheel
pip install dist/tellopy-*.dev*.whl --upgrade
```
then close everything and open a new terminal, in order to check if requirements have been installed:
```
conda activate <env>
pip freeze
```
Compare the list of packages installed to those in the file requirements.txt

For collision avoidance
```
mkdir saved_models
mkdir data
mkdir data/blocked
mkdir data/free
mkdir edges
mkdir edges/blocked_edges
mkdir edges/free_edges
```
For camera calibration
```
mkdir chessboards
```

## Control commands
All control commands are described in telloCV.py.

## Files
telloCV.py: controller

face_rec_tracker.py: uses a SVM and python's face_recognition to recognize faces, the binary SVM can be computed with "svm.py" DANGER!

svm.py: Creates a SVM capable of recognize faces, at the top of the file is shown how one should organize the images. [Face recognition](https://github.com/ageitgey/face_recognition)

collision_avoidance.py: It uses opencv canny edge detector to detect edges and uses the output as input for a neural network which performs collision avoidance, chooses whether to go forward or turn. DANGER!

train_model.ipynb: Can be used to train a NN using images in folder "data" and saving the NN in folder "saved_models". In order to save images look at the commands in telloCV.py

### Collision avoidance
In order to perform collision avoidance a neural network is required, train it with 'train_model.ipynb'.

The images can be acquired from telloCV.py ('f' for image to be labelled as 'free', 'b' for image to be labelled as 'blocked').

Once the NN has been trained and saved in folder 'saved_models' as best_model.pth, one can activate the collision avoidance feature by pressing 'q' while in telloCV.py.

IMPORTANT: At the moment only one between face recognition and collision avoidance can be active.

### Face recognition
The binarized svm in the repo, "svm_fam.bin", should be replaced with an svm fitted with your images, using the script "svm.py".

Instrunctions on how to organize the images are available in the script.

IMPORTANT: At the moment only one between face recognition and collision avoidance can be active.

### Camera Calibration
Save 15-20 images of a chessboard, made with the camera of tello, in the folder chessboards and call them <integer>.jpg.

Use the jupyter notebook 'camera_calibration' to compute the parameters and then copy and paste them in the python script 'image_processing.py'.
