# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 16:40:12 2019

This program downloads Object Detection model and loads to memory. 
OpenCY opens webcam.
The Model will now detect objects in each frame of live webcam video.
If a Car is detected in the live feed, trigger URL with yes parameter.
Once the car moves out of the frame, trigger URL with no parameter.

This program needs to download few packages. Refer below link for more details:
https://www.youtube.com/watch?v=wh7_etX91ls&t=739s

@author: Nirmal
"""

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
#tf.disable_v2_behavior()
import tensorflow.compat.v1 as tf1
import requests
#import zipfile
 
#from collections import defaultdict
#from io import StringIO
#from matplotlib import pyplot as plt
#from PIL import Image
 
import cv2

 
sys.path.append("..")
 
from utils import label_map_util
 
from utils import visualization_utils as vis_util
 
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
 
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
 
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
 
NUM_CLASSES = 90
 
opener = urllib.request.URLopener()
opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
  file_name = os.path.basename(file.name)
  if 'frozen_inference_graph.pb' in file_name:
    tar_file.extract(file, os.getcwd())
 
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf1.GraphDef()
  tf.gfile = tf.io.gfile    #bnk
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')
 
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)
 
with detection_graph.as_default():
  with tf1.Session(graph=detection_graph) as sess:
    cap = cv2.VideoCapture(0)
    urlsent = False    
    nocarflips = 0
    carflips = 0
    while True:
     ret, image_np = cap.read()
     if ret == False:
         print ('unable to load webcam')
         break
     image_np = cv2.flip(image_np,1)      #flip the image from webcam video
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
     image_np_expanded = np.expand_dims(image_np, axis=0)
     image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
     boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
     scores = detection_graph.get_tensor_by_name('detection_scores:0')
     classes = detection_graph.get_tensor_by_name('detection_classes:0')
     num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    # Actual detection.
     (boxes, scores, classes, num_detections) = sess.run(
       [boxes, scores, classes, num_detections],
       feed_dict={image_tensor: image_np_expanded})
    # Visualization of the results of a detection.
     vis_util.visualize_boxes_and_labels_on_image_array(
         image_np,
         np.squeeze(boxes),
         np.squeeze(classes).astype(np.int32),
         np.squeeze(scores),
         category_index,
         use_normalized_coordinates=True,
         line_thickness=8)
 
     cv2.imshow('object detection', cv2.resize(image_np, (800,600)))
     
    # Get the percentage value for given id. In this case, it is a car.
    # If car is present in frame for 15 flips, send a url with status as yes
     output = [category_index.get(value) for index,value in enumerate(classes[0]) if scores[0,index] > 0.5]     
     carfound = False
     for i in range(len(output)):
         if output[i]['id'] == 3:                 # id 3 is for car
             object = output[i]['name']
             carfound = True
             carflips += 1
             print (object + ' found flips - ' + str(carflips))
             nocarflips = 0
             if carflips >= 8 and urlsent == False:
                 url = 'https://digital-mrkt.tpfsoftware.com/iTrack/iot_device?device_type=image&detect=car&device_status=yes'
                 res = requests.get(url)
                 urlsent = True
                 carflips = 0
                 if res.status_code == 200:
                     print('yes msg sent')
                     print(res.text)
                 else:
                     print('yes msg something went wrong')
                     res = requests.get(url)
                        
         
     if carfound == False:                         #if car is not present in frame
         carflips = 0

    # If car is not present in frame for 15 flips and yes staus msg is sent, send a url with status as no
     if urlsent == True and carfound == False:
         nocarflips += 1
         print(object + ' not found flips - ' + str(nocarflips))
         if nocarflips > 15:
             url = 'https://digital-mrkt.tpfsoftware.com/iTrack/iot_device?device_type=image&detect=car&device_status=no'
             res = requests.get(url)
             if res.status_code == 200:
                 print('no msg sent')
                 print(res.text)
             else:
                 print('no something went wrong')
                 res = requests.get(url)
             carfound = False
             urlsent = False
             nocarflips = 0
         
    # release opencv and close windows            
     if cv2.waitKey(1) & 0xFF == ord('q'):
       cap.release()
       cv2.destroyAllWindows()
       break