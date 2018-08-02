# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 14:27:15 2018

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
from Detectionv4 import Detectionv4
from Driver_failsv5 import Driver_failsv5
from speed_extractionv3 import speed_extractionv3
from speed_analysisv2 import speed_analysisv2
# Name of the directory containing the object detection module we're using
MODEL_NAME = 'inference_graph'
# Grab path to current working directory
CWD_PATH = os.getcwd()
folder='worst cases\\Bad\\'
#folder='better cases\\Good\\'
file='1'
CWD_PATH_IMG = 'C:\\tensorflow1\\models\\research\\object_detection\\test_images\\'+folder+file
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
frame_init='none'
frame_end='none'
fails_buffer=[] #buffer with fails detected in all the frames
speed_buffer=[] #buffer with vehicle speed during detection
detect_buffer=[]
true_det=[] #vector with true signal detected 
counter=[0,0,0,0,0,0,0,0] #vector used as counter of positive detection per frame, considering only 8 important labels
counter2=[0,0,0,0,0,0,0,0] #vector used as counter to wax elements when a long time has passed without detections
location_x=[0,0,0,0,0,0,0,0] #vector with location in X of objects in truth detection 
fail='none' #label used to graph indicators of fails
speed_change='none'  #label used to graph indicators of fails
signal_det='none' #label used to graph indicators of fails
limit=3 #number of repetitions requiered to accept a detection
counter3=0 #to clear the fail window after a few time without detection
counter4=0 #to mantain the detection alert after no detection
true_det_past=['none']
while frame<= N :
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
    
    #function to ensure the true-detection of signals
    [true_det,counter,counter2,location_x]=Detectionv4(class_detected,box_detected,true_det,counter,counter2,location_x,limit,true_det_past)
    print ('COUNTERS: rojo:%d amarillo:%d verde:%d pare:%d cruce:%d cebra:%d ceda:%d peaton:%d'%(counter[0],counter[1],counter[2],counter[3],counter[4],counter[5],counter[6],counter[7]))       
    print (true_det)
    print (true_det_past)
    #this is done when at least a true detection is assured
    if true_det!=[]: 
        if frame_init=='none':
            frame_init=frame
        #fail=true_det[0] #to clear the fail window
        repeated=0 #is reset in each frame
        for k in true_det:
            for l in class_detected:
                if k==l: 
                    repeated=1
        #To add an extra time, waiting for a new detection of a possible traffic light 
        if true_det[0]!='amarillo' and true_det[0]!='verde':
            if repeated==0:
                counter4=counter4+1
                print(counter4)
            if counter4==4*limit:
                counter4=0
            else:
                repeated=1                      
        #if signals previously detected now are not detected, then continue with the next step:
        if repeated==0:  
            frame_end=frame
            #function to extract the speed record
            speed_buffer=speed_extractionv3(frame_init,frame_end,N,folder,file)
            #function to estimate the speed changes (input:speed_buffer)
            speed_change=speed_analysisv2(speed_buffer) #outputs: aumenta, disminuye, igual, detiene
            #function to infer the driver behavior
            fail=Driver_failsv5(speed_change,true_det) #outputs: rojo,amarillo,verde 
            signal_det=true_det[0]
            #buffers to be used for a later analysis
            fails_buffer=fails_buffer+[fail]
            detect_buffer=detect_buffer+[true_det]
            print('ANÁLISIS: True Detection:%s Speed Change: %s Fail:%s'%(true_det,speed_change,fail))
            #wax true_det, elementes in counter greater than 4 and speed_buffer
            true_det_past=true_det
            true_det=[]
            speed_buffer=[]
            frame_init='none'
            frame_end='none'
            counter3=0
            for k in range(len(counter)):
               if counter[k]>=limit:
                    counter[k]=0
            counter[0]=0
            counter[1]=0
            counter[2]=0
    #to erase the fail window
    elif true_det==[] and counter3>=4*limit:
        true_det_past=['none']
        fail='none'
        speed_change='none'
        signal_det='none'
    counter3=counter3+1
    #INDICATORS       
    if signal_det=='none':
        indicator = 'nada.jpg'
    elif signal_det=='rojo':
        indicator = 'semaforo_rojo.jpg'
    elif signal_det=='amarillo':
        indicator = 'semaforo_amarillo.jpg'
    elif signal_det=='verde':
        indicator = 'semaforo_verde.jpg'
    elif signal_det=='pare':
        indicator = 'pare.jpg'
    elif signal_det=='cruce' or signal_det=='cebra':
        indicator = 'cruce.jpg'
    elif signal_det=='ceda':
        indicator = 'ceda.jpg'
    PATH_TO_INDICATORS = os.path.join(CWD_PATH,'mensajes',indicator)
    indicator_detection = cv2.imread(PATH_TO_INDICATORS)
    if fail=='t_roja':
        indicator = 'mal.jpg'
    elif fail=='t_verde':
        indicator = 'bien.jpg'
    elif fail=='none':
        indicator = 'nada.jpg'
    PATH_TO_INDICATORS = os.path.join(CWD_PATH,'mensajes',indicator)
    indicator_fail = cv2.imread(PATH_TO_INDICATORS)
    if speed_change=='aumenta':
        indicator = 'aumento.jpg'
    elif speed_change=='disminuye':
        indicator = 'disminuye.jpg'
    elif speed_change=='detiene':
        indicator = 'detuvo.jpg'
    elif speed_change=='igual':
        indicator = 'igual.jpg'
    elif speed_change=='none':
        indicator = 'nada.jpg'
    PATH_TO_INDICATORS = os.path.join(CWD_PATH,'mensajes',indicator)
    indicator_speed = cv2.imread(PATH_TO_INDICATORS)
    PATH_TO_INDICATORS = os.path.join(CWD_PATH,'mensajes','plantilla.jpg')
    indicator_behavior = cv2.imread(PATH_TO_INDICATORS)
    image[355:355+indicator_behavior.shape[0], 0:0+indicator_behavior.shape[1]] = indicator_behavior
    image[378:378+indicator_fail.shape[0], 611:611+indicator_fail.shape[1]] = indicator_fail
    image[378:378+indicator_detection.shape[0], 55:55+indicator_detection.shape[1]] = indicator_detection
    image[378:378+indicator_speed.shape[0], 322:322+indicator_speed.shape[1]] = indicator_speed
    a=indicator_fail.shape[0]
    b=indicator_detection.shape[0]
    c=indicator_behavior.shape[0]
    d=image.shape[1]
    cv2.namedWindow('VIDEO', cv2.WINDOW_NORMAL)
    cv2.imshow('VIDEO', image)
    
    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break
  # print ("frame : %d " % frame)
    frame=frame+1
    
    # Press any key to close the image
cv2.waitKey(0)

# Clean up
cv2.destroyAllWindows()
