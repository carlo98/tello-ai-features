# Tello AI Features
Autonomous tracker for Tello drones. It works on python 3.7 and 3.6 and uses [DJITelloPy](https://github.com/damiafuentes/DJITelloPy).

The starting point is [Ubotica telloCV](https://github.com/Ubotica/telloCV), Nov 6 2018.

The initial idea was to code a face recognition and tracker, then a lot of ideas have arrived and still are coming, so this repository will continue to grow, with the goal of providing a package of AI algorithms for Tello drones.

# Table of contents
1. [Installation](#installation)
2. [Control Commands](#control)
3. [Collision Avoidance](#ca)
    1. [Reinforcement Learning](#rl)
4. [Face Recognition](#fr)
5. [Go Back to Origin](#loc)
6. [Camera Calibration](#cc)
7. [Files Description](#f)
8. [References](#ref)

## Installation <a name="installation"></a>
Install anaconda and opencv and then:
```
conda create --name <env> python=3.7 # or python=3.6
conda activate <env>
conda install pip
pip install numpy av pynput face-recognition sklearn torch jupyter
```

djitellopy:
```
pip install https://github.com/damiafuentes/DJITelloPy/archive/master.zip
```

For collision avoidance, in repository's root folder do:
```
mkdir Collision_Avoidance/saved_models
mkdir Collision_Avoidance/data
mkdir Collision_Avoidance/data/blocked
mkdir Collision_Avoidance/data/free
mkdir Collision_Avoidance/saliency
mkdir Collision_Avoidance/saliency/blocked
mkdir Collision_Avoidance/saliency/free
```

For RL training, in repository's root folder do:
```
mkdir Collision_Avoidance/rl_saved_models
```

For face recognition, in repository's root folder do:
```
mkdir Face_Recognition/train_dir
```

For camera calibration, in repository's root folder do:
```
mkdir Camera_Calibration/chessboards
```

## Control commands <a name="control"></a>
All control commands are described in telloCV.py.

For features:
- '1': To toggle collision avoidance
- '2': To toggle face recognition
- '3': To toggle reinforcement learning training
- '4': To toggle go back to origin
- 'x': To end/start episode of RL
- 'f': To take a picture and label it as 'free'
- 'b': To take a picture and label it as 'blocked'

## Collision avoidance <a name="ca"></a>
From repository's root folder:
```
conda activate <env>
python3 telloCV.py
```
In order to start/stop press '1'.

The images can be acquired from telloCV.py ('f' for image to be labelled as 'free', 'b' for image to be labelled as 'blocked').

Once the NN has been trained and saved in folder 'Collision_Avoidance/saved_models' as best_model.pth.

The NN provided in 'Collision_Avoidance/saved_models' has not been fully trained so pay attention please, it's there only to provide a starting point for transfer learning.

In order to provide fast inference for collision avoidance also in pc without GPU a small NN is provided, it accepts in input the saliency maps computed from raw images; this approach has led to a reduction in compution time of approximately 1/2 - 2/3 on intel-i5, with respect to AlexNet.

The drawback is the amount of training images required, due to a partial loss of transfer learning weights.

In order to perform collision avoidance a neural network is required, train it with 'Collision_Avoidance/train_model.ipynb', it needs to be launched from 'Collision_Avoidance' folder.

From Collision_Avoidance folder:
```
conda activate <env>
jupyter notebook
```

IMPORTANT: At the moment only one between face recognition and collision avoidance can be active.

### Reinforcement Learning Training <a name="rl"></a>
From repository's root folder:
```
conda activate <env>
python3 telloCV.py
```
In order to start/stop press '3'.

It is possible to further train the collision avoidance model with online reinforcement learning, this relies on the user to detect collisions by pressing 'x'. If no collision is detected by the user each episode will terminate after a given amount of steps (default: 100, change in 'Collision_Avoidance/RL.py').

Every time an episode ends the drone stops, giving you the time to move it to another position while the network is training; in order to restart inference press 'x' a second time.

Reward: +1/max_steps_per_episode if agent decides to go forward, 0 if it turns, -1 for collisions.

Do not attempt to train a model from scratch with this method because it requires a lot of time and it would seem nearly impossible, first get a collision avoidance model trained with 'train_model.pynb' as satisfactory as possible and then proceed with this.

The model trained by RL is saved into the folder 'Collision_Avoidance/rl_saved_models'.

## Face recognition <a name="fr"></a>
From repository's root folder:
```
conda activate <env>
python3 telloCV.py
```
In order to start/stop press '2'.

The binarized svm in the repo, "Face_Recognition/svm_fam.bin", should be replaced with an svm fitted with your images, using the script "Face_Recognition/svm.py".
From Face Recognition folder:
```
conda activate <env>
python3 svm.py
```

Instructions on how to organize the images are available in the script.

By changing the two parameters at the beginning of the python script 'Face_Recognition/face_rec_tracker.py' one can choose which person the tello should track and the ratio between recall and speed, for face detection and recognition.

IMPORTANT: At the moment only one between face recognition and collision avoidance can be active.

## Go Back to Origin <a name="loc"></a>
From repository's root folder:
```
conda activate <env>
python3 telloCV.py
```
Always on, in order to go back to origin press '4'.

At the moment a lot of hypotesis are made in order to simplify the setting, but in future will be removed one at a time.

For now it works only with manual controls.

## Camera Calibration <a name="cc"></a>
Save 15-20 images of a chessboard, made with the camera of tello, in the folder 'Camera_Calibration/chessboards' and call them n.jpg, (n=0, 1, ...).

Use the jupyter notebook 'Camera_Calibration/camera_calibration.ipynb' to compute the parameters and then copy and paste them in the python script 'Camera_Calibration/process_image.py'.

From Camera Calibration folder:
```
conda activate <env>
jupyter notebook
```

## Files Description <a name="f"></a>
telloCV.py: controller

Face_Recognition/face_rec_tracker.py: uses a SVM and python's face_recognition to recognize faces, the binary SVM can be computed with "svm.py" DANGER!

Face_Recognition/svm.py: Creates a SVM capable of recognize faces, at the top of the file is shown how one should organize the images. [Face recognition](https://github.com/ageitgey/face_recognition)

Collision_Avoidance/collision_avoidance.py: It uses the saliency map generator provided [here](https://github.com/tobybreckon/DoG-saliency) to generate lower dimensional inputs to use in a neural network which performs collision avoidance, chooses whether to go forward or turn. DANGER!

Collision_Avoidance/train_model.ipynb: Can be used to train a NN using images in folder "data" and saving the NN in folder "saved_models". In order to save images look at the commands in telloCV.py

Camera_Calibration/camera_calibration.ipynb: Can be used to compute the camera parameters.

Camera_Calibration/process_image.py: Provides a class which computes undistorted images, given camera parameters.

Collision_Avoidance/model.py: NN model.

Collision_Avoidance/RL.py: Reinforcement learning script, change parameters in the 'init' method to change saving/training frequencies and other RL parameters.

Localization/pose_estimation.py: Provides a class that keeps track of ideal position with respect to starting point, at the moment a lot of hypotesis are used.

## References <a name="ref"></a>

If you are making use of this work in any way please reference the following articles in any report, publication, presentation, software release or any other associated materials:

[Real-time Visual Saliency by Division of Gaussians](https://breckon.org/toby/publications/papers/katramados11salient.pdf)
(Katramados, Breckon), In Proc. International Conference on Image Processing, IEEE, 2011.
```
@InProceedings{katramados11salient,
  author    =    {Katramados, I. and Breckon, T.P.},
  title     = 	 {Real-time Visual Saliency by Division of Gaussians},
  booktitle = 	 {Proc. Int. Conf. on Image Processing},
  pages     = 	 {1741-1744},
  year      = 	 {2011},
  month     = 	 {September},
  publisher =    {IEEE},
  url       = 	 {https://breckon.org/toby/publications/papers/katramados11salient.pdf},
  doi       = 	 {10.1109/ICIP.2011.6115785},
}
```

For non-commercial use (i.e. academic, non-for-profit and research) the (very permissive) terms of the MIT free software [LICENSE](LICENSE) must be adhered to.

For commercial use, the Division of Gaussians (DIVoG / DoG) saliency detection algorithm is patented (WIPO reference: [WO2013034878A3](https://patents.google.com/patent/WO2013034878A3/)) and available for licensing via [Cranfield University](https://www.cranfield.ac.uk/).
