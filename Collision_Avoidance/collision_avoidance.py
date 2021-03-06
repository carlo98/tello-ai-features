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
from Collision_Avoidance.model import tommy_net
from Collision_Avoidance.saliency_map import SaliencyDoG

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
        _, frame = tracker.track(np.array(frame))
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
        self.model = tommy_net()
        self.model.load_state_dict(torch.load('Collision_Avoidance/saved_models/best_model.pth', map_location=torch.device('cpu')))

        self.device = torch.device('cpu')
        self.model = self.model.to(self.device)
        
        self.saliency_mapper = SaliencyDoG(pyramid_height=5, shift=5, ch_3=False,
                              low_pass_filter=True, multi_layer_map=False)
        
    def preprocess(self, x):
        x = self.saliency_mapper.generate_saliency(x)
        x = cv2.medianBlur(x, 9) # Reduce impulse noise
        x = cv2.GaussianBlur(x, (3, 3), 0.5) # Reduce linear noise
        y = torch.from_numpy(x.get()).float()
        y = y.to(self.device)
        y = y[None, None, ...]
        return y, x

    def track(self, frame):
        """NN Tracker"""
        #start_time = time.time()
        y, x = self.preprocess(frame)
        y = self.model(y)
        
        #print("Inference time: ", time.time()-start_time)
    
        # we apply the `softmax` function to normalize the output vector so it sums to 1 (which makes it a probability distribution)
        y = F.softmax(y, dim=1)
    
        #print(y)
        prob_blocked = float(y.flatten()[0])
    
        if prob_blocked < 0.30:
            return 0, x # forward
        else:
            #print("blocked")
            return 1, x # turn

if __name__ == '__main__':
    main()
