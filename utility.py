"""
Utility methods.

Author: Carlo Cena
"""
from scipy.interpolate import interp1d
import numpy as np
import imutils
import cv2


def interpolate_readings(raw_readings):
    """
    Predicts next position of target
    """
    readings = []
    readings_index = []
    flag = True  # Set to false if last reading has no face
    for i, reading in enumerate(raw_readings):
        if reading[2] != 0:
            readings.append(reading)
            readings_index.append(i)
        elif i == len(raw_readings)-1:
            flag = False

    if len(readings) >= 2:
        readings = np.array(readings)
        fx = interp1d(readings_index, readings[:, 0], fill_value="extrapolate")
        fy = interp1d(readings_index, readings[:, 1], fill_value="extrapolate")
        farea = interp1d(readings_index, readings[:, 2], fill_value="extrapolate")
        return fx(len(raw_readings)), fy(len(raw_readings)), farea(len(raw_readings))

    # If only one reading available using it only if it is the most recent one
    if len(readings) == 1 and flag:
        return readings[0][0], readings[0][1], readings[0][2]

    return -1, -1, -1


def get_frame(vid_stream, stream):
    """
    Grabs the current video frame
    """
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
    """
    Shows the frame to cv2 window
    """
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        exit()
