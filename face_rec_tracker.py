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

# import the necessary packages
import argparse
import time
import cv2
import imutils
from imutils.video import VideoStream
import pickle
import face_recognition
from sklearn import svm
import numpy as np

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
        tracker.track(frame)
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
    A cnn tracker and face recognition, it will look for object and
    create an x and y offset valuefrom the midpoint
    """

    def __init__(self):
        # Load SVM binarized
        with open("svm_fam.bin", "rb") as f:
            self.clf = pickle.load(f)
        
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
        start_time = time.time()
        process_this_frame = True
        # Resize frame of video to 1/2 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.50, fy=0.50)
            
        # Only process every other frame of video to save time
        if process_this_frame:
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(small_frame, model='ccn')
            face_encodings = face_recognition.face_encodings(small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                name = self.clf.predict([face_encoding])

                face_names.append(*name)
            print("Inference time: ", time.time()-start_time)
        #process_this_frame = not process_this_frame


        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= int(2)
            right *= int(2)
            bottom *= int(2)
            left *= int(2)

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
       
            x = (right-left)/2
            y = (top-bottom)/2
            #radius = np.max([x, y])
            x_c = int(x+left)
            y_c = int(y+bottom)
            
            cv2.circle(frame, (x_c, y_c), 2, (0, 0, 255), 2)

            self.xoffset = int(x_c - self.midx)
            self.yoffset = int(self.midy - y_c)
            area = (4*x*y)
        if len(face_names) == 0:
            self.xoffset = 0
            self.yoffset = 0
            area = 0
        # Display the resulting image
        self.draw_arrows(frame)
        show(frame)
        self.previous_detection.append((self.xoffset, self.yoffset, area))
        self.previous_detection.pop(0)
        print(frame.shape)
        print((self.xoffset, self.yoffset, area))
        print((self.midx, self.midy))
        return self.previous_detection

if __name__ == '__main__':
    main()
