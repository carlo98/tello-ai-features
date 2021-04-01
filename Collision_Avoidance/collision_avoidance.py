"""
Collision avoidance with NN
"""
import argparse
from imutils.video import VideoStream
import torch
import cv2
import numpy as np
import torch.nn.functional as F
import time
import random
from Collision_Avoidance.model import tommy_net
from Collision_Avoidance.saliency_map import SaliencyDoG
from utility import show, get_frame


def main():
    """
    Handles input from file or stream, tests the tracker class
    """
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


class Agent:

    def __init__(self):
        self.model = tommy_net()
        self.model.load_state_dict(
            torch.load('Collision_Avoidance/saved_models/best_model.pth', map_location=torch.device('cpu')))

        self.device = torch.device('cpu')
        self.model = self.model.to(self.device)

        self.saliency_mapper = SaliencyDoG(pyramid_height=5, shift=5, ch_3=False,
                                           low_pass_filter=True, multi_layer_map=False)
        self.last_move = 1

    def preprocess(self, x):
        x = self.saliency_mapper.generate_saliency(x)
        x = cv2.medianBlur(x, 9)  # Reduce impulse noise
        x = cv2.GaussianBlur(x, (3, 3), 0.5)  # Reduce linear noise
        y = torch.from_numpy(x.get()).float()
        y = y.to(self.device)
        y = y[None, None, ...]
        return y, x

    def track(self, frame):
        """
        NN Tracker
        """
        # start_time = time.time()
        y, x = self.preprocess(frame)
        y = self.model(y)

        # print("Inference time: ", time.time()-start_time)

        # we apply the `softmax` function to normalize the output vector so it sums to 1 (which makes it a probability distribution)
        y = F.softmax(y, dim=1)

        # print(y)
        prob_blocked = float(y.flatten()[0])

        if prob_blocked < 0.30:
            self.last_move = 0
            return 0, x  # forward
        else:
            if self.last_move == 0:
                self.last_move = random.randint(1, 2)  # Left or right at random if last move was going forward
                return self.last_move, x
            # print("blocked")
            return self.last_move, x  # If it was turning keeps turning in the same way


if __name__ == '__main__':
    main()
