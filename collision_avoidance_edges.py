"""
Collision avoidance with NN
"""
import argparse
from imutils.video import VideoStream
import imutils
import torch
import torchvision
import cv2
import numpy as np
import torch.nn.functional as F
import time

def main():
    """Handles inpur from file or stream, tests the tracker class"""
    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument("-v", "--video",
                           help="path to the (optional) video file")
    args = vars(arg_parse.parse_args())

    # if a video path was not supplied, grab the reference
    # to the webcam
    if not args.get("video", False):
        vid_stream = VideoStream(src=0).start()

    # otherwise, grab a reference to the video file
    else:
        vid_stream = cv2.VideoCapture(args["video"])

    # allow the camera or video file to warm up
    time.sleep(2.0)
    stream = args.get("video", False)
    frame = get_frame(vid_stream, stream)
    height, width = frame.shape[0], frame.shape[1]
    tracker = Agent()

    # keep looping until no more frames
    more_frames = True
    while more_frames:
        if tracker.track(np.array(frame)) == 1:
            print("Obstacle")
        frame = get_frame(vid_stream, stream)
        if frame is None:
            more_frames = False

    # if we are not using a video file, stop the camera video stream
    if not args.get("video", False):
        vid_stream.stop()

    # otherwise, release the camera
    else:
        vid_stream.release()

    # close all windows
    cv2.destroyAllWindows()


def get_frame(vid_stream, stream):
    """grab the current video frame"""
    frame = vid_stream.read()
    # handle the frame from VideoCapture or VideoStream
    frame = frame[1] if stream else frame
    # if we are viewing a video and we did not grab a frame,
    # then we have reached the end of the video
    if frame is None:
        return None
    else:
        frame = imutils.resize(frame, width=224, height=224)
        return frame


def show(frame):
    """show the frame to cv2 window"""
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        exit()


class Agent:

    def __init__(self):
        self.model = torchvision.models.alexnet(pretrained=False)
        self.model.features[0] = torch.nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2)
        self.model.classifier[6] = torch.nn.Linear(self.model.classifier[6].in_features, 2)
        self.model.load_state_dict(torch.load('saved_models/best_model.pth', map_location=torch.device('cpu')))

        self.device = torch.device('cpu')
        self.model = self.model.to(self.device)

        mean = 255.0 * np.array([0.485, 0.456, 0.406])
        stdev = 255.0 * np.array([0.229, 0.224, 0.225])

        self.resize = torchvision.transforms.Resize((224, 224))
        
    def preprocess(self, x):
        x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
        x = cv2.Canny(x, 20, 160)
        show(x)
        x = torch.from_numpy(x).float()
        x = x.to(self.device)
        x = x[None, None, ...]
        return x

    def track(self, frame):
        """NN Tracker"""
        start_time = time.time()
        x = self.preprocess(frame)
        y = self.model(x)
        
        print("Inference time: ", time.time()-start_time)
    
        # we apply the `softmax` function to normalize the output vector so it sums to 1 (which makes it a probability distribution)
        y = F.softmax(y, dim=1)
    
        print(y)
        prob_blocked = float(y.flatten()[0])
    
        if prob_blocked < 0.40:
            return 0 # forward
        else:
            return 1 # turn

if __name__ == '__main__':
    main()
