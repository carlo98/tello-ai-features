"""
Change parameters by taking the values from camera_calibration.ipynb

Those provided here are taken from the camera of my Tello.
"""
import cv2
import numpy as np

CAMERA_MATRIX = np.array([[903.06307185, 0.0, 641.01141885],
                          [0.0, 904.90985547, 488.70094853],
                          [0.0, 0.0, 1.0]], np.float32)

DISTORTION_COEFFICIENTS = np.array([7.48889296e-02, -5.10446633e-01, -1.50399483e-03, -2.01938062e-03, 2.04965398e+00],
                                   np.float32)

EXTRINSIC_MATRIX = np.array([[9.86843654e-01, -3.04127451e-02, 1.58791271e-01, -6.13350749e+01],
                             [5.36261926e-03, 9.87765217e-01, 1.55856080e-01, -4.35603859e+01],
                             [-1.61588505e-01, -1.52954046e-01, 9.74932928e-01, 3.10567114e+02]], np.float32)


class FrameProc:
    """
    This class is used to get undistorted images from tello.
    """

    def __init__(self, w, h):
        """
        w: width
        h: height
        """
        # Finding the new optical camera matrix
        newcameramtx, self.roi = cv2.getOptimalNewCameraMatrix(CAMERA_MATRIX, DISTORTION_COEFFICIENTS, (w, h), 1,
                                                               (w, h))

        # Getting the mapping between undistorted and distorted images
        self.mapx, self.mapy = cv2.initUndistortRectifyMap(CAMERA_MATRIX, DISTORTION_COEFFICIENTS, None, newcameramtx,
                                                           (w, h), 5)

    def undistort_frame(self, frame):
        """
        Gets the distorted frame and returns an undistorted one.
        """
        # Apply the mapping
        frame_undistorted = cv2.remap(frame, self.mapx, self.mapy, cv2.INTER_LINEAR)

        x, y, w_2, h_2 = self.roi
        frame_undistorted = frame_undistorted[y:y + h_2, x:x + w_2]

        return frame_undistorted
