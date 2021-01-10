"""
A tracker class for controlling the Tello and some sample code for showing how
it works. you can test it using your webcam or a video file to make sure it works.

it computes a vector of the ball's direction from the center of the
screen. The axes are shown below (assuming a frame width and height of 600x400):
+y                 (0,200)


Y  (-300, 0)        (0,0)               (300,0)


-Y                 (0,-200)
-X                    X                    +X

Based on the tutorial:
https://www.pyimagesearch.com/2015/09/14/ball-tracking-with-opencv/

Usage:
for existing video:
python tracker.py --video ball_tracking_example.mp4
For live feed:
python tracking.py

@author Leonie Buckley and Jonathan Byrne
@copyright 2018 see license file for details
"""

### YOLO5 ###

import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, non_max_suppression, apply_classifier, scale_coords, \
    xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
import numpy as np

# Initialize
set_logging()
device = select_device('cpu')
half = device.type != 'cpu'  # half precision only supported on CUDA

IMG_S = 416
WEIGHTS = "models_my/me/best.pt"
CONF_THRES = 0.50
IOU_THRES = 0.45
CLASSES = 3

### END YOLO5 ###

# import the necessary packages
import argparse
import time
import cv2
import imutils
from imutils.video import VideoStream

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
    tracker = Tracker()
    tracker.init_video(height, width)

    # keep looping until no more frames
    more_frames = True
    while more_frames:
        frame = cv2.resize(frame, (IMG_S, IMG_S))
        tracker.track(frame)
        frame = tracker.draw_arrows(frame)
        show(frame)
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
        frame = imutils.resize(frame, width=600)
        return frame


def show(frame):
    """show the frame to cv2 window"""
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        exit()


class Tracker:
    """
    A cnn tracker, it will look for object and
    create an x and y offset valuefrom the midpoint
    """

    def __init__(self):
        # Load model
        self.model = attempt_load(WEIGHTS, map_location=device)  # load FP32 model
        self.imgsz = check_img_size(IMG_S, s=self.model.stride.max())  # check img_size
        if half:
            self.model.half()  # to FP16
            
        # Get names and colors
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]
        
        img = torch.zeros((1, 3, self.imgsz, self.imgsz), device=device)  # init img
        _ = self.model(img.half() if half else img) if device.type != 'cpu' else None  # run once

        
    def init_video(self, height, width):
        self.height = height
        self.width = width
        self.midx = int(width / 2)
        self.midy = int(height / 2)
        self.xoffset = 0
        self.yoffset = 0
        self.previous_detection = [(0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)]

    def draw_arrows(self, frame):
        """Show the direction vector output in the cv2 window"""
        #cv2.putText(frame,"Color:", (0, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, thickness=2)
        cv2.arrowedLine(frame, (self.midx, self.midy),
                        (self.midx + self.previous_detection[-1][0], self.midy - self.previous_detection[-1][1]),
                        (0, 0, 255), 5)
        return frame

    def track(self, frame):
        """NN Tracker"""
        img = np.reshape(frame, (1, 3, self.imgsz, self.imgsz))
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
            
        # Inference
        start_time = time.time()
        detections = self.model(img)[0]
        print(detections)
        # Apply NMS
        detections = non_max_suppression(detections, CONF_THRES, IOU_THRES, classes=CLASSES)#, agnostic=opt.agnostic_nms)
        print("Inference time: ",time.time()-start_time)
        
        
        for i, det in enumerate(detections):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img.shape).round()
            ymin, xmin, ymax, xmax = detections
            ymin *= self.height
            ymax *= self.height
            xmin *= self.width
            xmax *= self.width
            x = (xmax-xmin)/2
            y = (ymax-ymin)/2
            radius = np.max([(xmax-xmin)/2, (ymax-ymin)/2])
            x = x+xmin
            y = y+ymin
            
            # draw the circle and centroid on the frame
            cv2.circle(frame, (int(x), int(y)), int(radius),
                       (0, 255, 255), 2)
            cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)

            self.xoffset = int(x - self.midx)
            self.yoffset = int(self.midy - y)
        else:
            self.xoffset = 0
            self.yoffset = 0
            radius = -1
        self.previous_detection.append((self.xoffset, self.yoffset, radius))
        self.previous_detection.pop(0)
        return self.previous_detection

if __name__ == '__main__':
    main()
