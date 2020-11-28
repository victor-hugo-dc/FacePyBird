import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import math
from scipy.signal import savgol_filter
from scipy import stats
import pandas as pd
from datetime import timedelta, datetime


class FaceDetector:
    """Detect human face from image"""

    def __init__(self, dnn_proto_text='models/deploy.prototxt', dnn_model='models/res10_300x300_ssd_iter_140000.caffemodel'):
        tf.get_logger().setLevel('ERROR')
        self.face_net = cv2.dnn.readNetFromCaffe(dnn_proto_text, dnn_model) # load the network and pass the model's configuration as its arguments
        self.detection_result = None # initialize the result to None

    def get_faceboxes(self, image, threshold=0.5):
        # get the bounding box of faces in image using dnn.
        rows, cols, _ = image.shape # get rows and columns of the image
        confidences = [] # confidence list
        faceboxes = [] # list of lists containing the coordinates of the facebox that was found

        # self.face_net is the saved model we loaded from memory
        # cv2.dnn.blobFromImage(image, scale_factor, size, mean, swapRB, crop)
        # creates a 4-dimensional blob from image. optionally resizes and crops the input image from center, subtracts mean values,
        # scales values by scale_factor, and then swaps Blue and Red channels if specified.
        self.face_net.setInput(cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0), False, False)) # set the 4-dimensional blob from the image as the input to our network
        detections = self.face_net.forward() # use the neural network to detect faces

        for result in detections[0, 0, :, :]: # loops through the faces detected
            confidence = result[2] # get the confidence of the model's detection
            if confidence > threshold: # if the confidence of the model is greater than our confidence threshold (50%)
                x_left_bottom, x_right_top = result[3] * cols, result[5] * cols # scales the output of the network to the columns of the image
                y_left_bottom, y_right_top = result[4] * rows, result[6] * rows # scales the output of the network to the rows of the image

                face_boxes = list(map(lambda x: int(x), [x_left_bottom, y_left_bottom, x_right_top, y_right_top])) # int(x) every element in a specific order

                confidences.append(confidence) # append the confidence of the face detection result to the confidences list
                faceboxes.append(face_boxes) # append the faceboxes we got from the model

        self.detection_result = [faceboxes, confidences] # set the detection result to the pair of faceboxes, confidences

        return confidences, faceboxes # note the order in which the fuction returns values


class MarkDetector:
    """Facial landmark detector by Convolutional Neural Network"""

    def __init__(self, saved_model='models/pose_model'):
        """Initialization"""
        # A face detector is required for mark detection.
        self.face_detector = FaceDetector()
        self.cnn_input_size = 128 # convolutional neural network input size is 128
        self.marks = None

        # Restore model from the saved_model file.
        self.model = keras.models.load_model(saved_model)

    @staticmethod
    def draw_box(image, boxes, box_color=(255, 255, 255)):
        """Draw square boxes on image"""
        for box in boxes:
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), box_color, 3)

    @staticmethod
    def move_box(box, offset):
        """Move the box to direction specified by vector offset"""
        left_x, right_x = box[0] + offset[0], box[2] + offset[0] 
        top_y, bottom_y = box[1] + offset[1], box[3] + offset[1] 
        return [left_x, top_y, right_x, bottom_y]

    @staticmethod
    def get_square_box(box):
        """Get a square box out of the given box, by expanding it."""
        left_x, right_x = box[0], box[2]
        top_y, bottom_y = box[1], box[3]

        box_width = right_x - left_x
        box_height = bottom_y - top_y

        # Check if box is already a square. If not, make it a square.
        diff = box_height - box_width
        delta = int(abs(diff) / 2)

        if diff == 0:                   # Already a square.
            return box
        elif diff > 0:                  # Height > width, a slim box.
            left_x -= delta
            right_x += delta
            if diff % 2 == 1:
                right_x += 1
        else:                           # Width > height, a short box.
            top_y -= delta
            bottom_y += delta
            if diff % 2 == 1:
                bottom_y += 1

        # Make sure box is always square.
        assert ((right_x - left_x) == (bottom_y - top_y)), 'Box is not square.'

        return [left_x, top_y, right_x, bottom_y]

    @staticmethod
    def box_in_image(box, image):
        """Check if the box is in image"""
        # rows = image.shape[0]
        # cols = image.shape[1]
        rows, cols = image.shape[:2]
        return box[0] >= 0 and box[1] >= 0 and box[2] <= cols and box[3] <= rows

    def extract_cnn_facebox(self, image):
        """Extract face area from image."""
        _, raw_boxes = self.face_detector.get_faceboxes(image=image, threshold=0.65)
        a = []
        for box in raw_boxes:
            # Move box down.
            # diff_height_width = (box[3] - box[1]) - (box[2] - box[0])
            offset_y = int(abs((box[3] - box[1]) * 0.1))
            box_moved = self.move_box(box, [0, offset_y])

            # Make box square.
            facebox = self.get_square_box(box_moved)

            if self.box_in_image(facebox, image):
                a.append(facebox)

        return a

    def detect_marks(self, image_np):
        """Detect marks from image"""

        # # Actual detection.
        predictions = self.model.signatures["predict"](tf.constant(image_np, dtype=tf.uint8))

        # Convert predictions to landmarks.
        marks = np.array(predictions['output']).flatten()[:136]
        marks = np.reshape(marks, (-1, 2))
        
        return marks

    @staticmethod
    def draw_marks(image, marks, color=(255, 255, 255)):
        """Draw mark points on image"""
        for mark in marks:
            cv2.circle(image, (int(mark[0]), int(mark[1])), 2, color, -1, cv2.LINE_AA)
            
            
def draw_annotation_box(img, rotation_vector, translation_vector, camera_matrix, color=(255, 255, 0), line_width=2):
    """Draw a 3D box as annotation of pose"""
    point_3d = []
    dist_coeffs = np.zeros((4,1))
    rear_size = 1
    rear_depth = 0
    point_3d.append((-rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, -rear_size, rear_depth))

    front_size = img.shape[1]
    front_depth = front_size*2
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d.append((-front_size, front_size, front_depth))
    point_3d.append((front_size, front_size, front_depth))
    point_3d.append((front_size, -front_size, front_depth))
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d = np.array(point_3d, dtype=np.float).reshape(-1, 3)

    # Map to 2d img points
    (point_2d, _) = cv2.projectPoints(point_3d, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    point_2d = np.int32(point_2d.reshape(-1, 2))
    k = (point_2d[5] + point_2d[8])//2
    
    return (point_2d[2], k)