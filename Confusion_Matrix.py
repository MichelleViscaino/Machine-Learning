# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 11:32:04 2018

@author: Leonardo
"""

######## Image Object Detection Using Tensorflow-trained Classifier #########
# Import packages
#from PIL import ImageGrab
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util
from Test_Obj_Det import Test_Obj_Det

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'inference_graph'
# Grab path to current working directory
CWD_PATH = os.getcwd()
CWD_PATH_IMG = 'C:\\tensorflow1\\models\\research\\object_detection\\test_images\\confusion'
#CWD_PATH_IMG = 'C:\\tensorflow1\\models\\research\\object_detection\\images\\test'
IMAGE_NAME = os.listdir(CWD_PATH_IMG) #vector con el nombre de todas las imagenes
# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')
# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')
# Number of classes the object detector can identify
NUM_CLASSES = 8
# Load the label map.
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)
# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
    sess = tf.Session(graph=detection_graph)
# Define input and output tensors (i.e. data) for the object detection classifier

# Define variables used
N=len(IMAGE_NAME)-1 #number of images to detect
frame=0 #frame initialization
class_detected_buffer=[]
box_detected_buffer=[]
while frame<=N :
    # Input tensor is the image
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Output tensors are the detection boxes, scores, and classes
    # Each box represents a part of the image where a particular object was detected
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    # Each score represents level of confidence for each of the objects.
    # The score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    # Number of objects detected
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    # Load image using OpenCV and
    # expand image dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    PATH_TO_IMAGE = os.path.join(CWD_PATH_IMG,IMAGE_NAME[frame])
    image = cv2.imread(PATH_TO_IMAGE)
    image_expanded = np.expand_dims(image, axis=0)
    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_expanded})
    # Draw the results of the detection (aka 'visulaize the results')
    [image, class_detected,box_detected]=vis_util.visualize_boxes_and_labels_on_image_array(
        image,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8,
        min_score_thresh=0.80)
    
    class_detected_buffer=class_detected_buffer+[class_detected]
    box_detected_buffer=box_detected_buffer+[box_detected]
      
    cv2.namedWindow('VIDEO', cv2.WINDOW_NORMAL)
    cv2.imshow('VIDEO', image)
    
    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break
    print ("frame : %d " % frame)
    frame=frame+1
    
    # Press any key to close the image
cv2.waitKey(0)

Test_Obj_Det(class_detected_buffer,box_detected_buffer)

# Clean up
cv2.destroyAllWindows()

