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

#Tensorflow
import tensorflow as tf
import pathlib
import time
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')   # Suppress Matplotlib warnings
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
tf.get_logger().setLevel('ERROR')

# Download and extract model
def download_model(model_name, model_date):
    base_url = 'http://download.tensorflow.org/models/object_detection/tf2/'
    model_file = model_name + '.tar.gz'
    model_dir = tf.keras.utils.get_file(fname=model_name,
                                        origin=base_url + model_date + '/' + model_file,
                                        untar=True)
    return str(model_dir)

MODEL_DATE = '20200711'
MODEL_NAME = 'ssd_mobilenet_v2_320x320_coco17_tpu-8'
PATH_TO_MODEL_DIR = download_model(MODEL_NAME, MODEL_DATE)

# Download labels file
def download_labels(filename):
    base_url = 'https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/data/'
    label_dir = tf.keras.utils.get_file(fname=filename,
                                        origin=base_url + filename,
                                        untar=False)
    label_dir = pathlib.Path(label_dir)
    return str(label_dir)

LABEL_FILENAME = 'mscoco_label_map.pbtxt'
PATH_TO_LABELS = download_labels(LABEL_FILENAME)

PATH_TO_SAVED_MODEL = PATH_TO_MODEL_DIR + "/saved_model"

def load_model():
    print('Loading model...', end='')
    start_time = time.time()

    # Load saved model and build the detection function
    detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Done! Took {} seconds'.format(elapsed_time))
    
    return detect_fn


category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                    use_display_name=True)
                                                                

def detect(image_np, detect_fn):

    print('Running inference.')
    # Things to try:
    # Flip horizontally
    # image_np = np.fliplr(image_np).copy()

    # Convert image to grayscale
    # image_np = np.tile(
    #     np.mean(image_np, 2, keepdims=True), (1, 1, 3)).astype(np.uint8)

    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image_np)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # input_tensor = np.expand_dims(image_np, 0)
    detections = detect_fn(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                   for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = (detections['detection_classes'].astype(np.int64))
    #print(detections['detection_classes'])
    
    #Filtering on target
    indexes = detections['detection_classes'] == 1
    detections['detection_classes'] = detections['detection_classes'][indexes]
    detections['detection_boxes'] = detections['detection_boxes'][indexes]
    detections['detection_scores'] = detections['detection_scores'][indexes]
    
    #Only best one
    if len(detections['detection_scores'])>0:
        best_index = [np.argmax(detections['detection_scores'])]
        detections['detection_classes'] = detections['detection_classes'][best_index]
        detections['detection_boxes'] = detections['detection_boxes'][best_index]
        detections['detection_scores'] = detections['detection_scores'][best_index]

        #image_np_with_detections = image_np.copy()

        #viz_utils.visualize_boxes_and_labels_on_image_array(
    #      image_np_with_detections,
    #      detections['detection_boxes'],
    #      detections['detection_classes'],
    #      detections['detection_scores'],
    #      category_index,
    #      use_normalized_coordinates=True,
    #      max_boxes_to_draw=200,
    #      min_score_thresh=.30,
    #      agnostic_mode=False)
        #show(image_np_with_detections)
    
        return detections['detection_boxes'][0]
    return -1, -1, -1, -1
#

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
    tracker = Tracker(height, width)

    # keep looping until no more frames
    more_frames = True
    while more_frames:
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
        self.detect_fn = load_model()
        
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
        """Cnn tracker"""
        detections = detect(frame, self.detect_fn)

        # only proceed if at least one contour was found
        if detections[0] != -1:
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
