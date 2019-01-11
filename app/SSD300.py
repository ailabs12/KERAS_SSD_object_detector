#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 10:49:02 2018
SSD300 Inference by trained model
@author: kdg
"""

import numpy as np
import cv2
import base64
import io
from keras.preprocessing import image

# Set the image size.
img_height = 300
img_width = 300

classes = ['background',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat',
           'chair', 'cow', 'diningtable', 'dog',
           'horse', 'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor']
fltr = ['bus', 'car', 'train', 'truck']

def detector(image_body, img_header, model): 
    
    """ 2. Load some images """
    orig_images = [] # Store the images here.
    input_images = [] # Store resized versions of the images here.
    unchanged_image = cv2.imdecode(np.fromstring(image_body,
                           np.uint8), cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(unchanged_image, cv2.COLOR_BGR2RGB)
    orig_images.append(img) # # Store the original image
    img = cv2.resize(img, (img_width, img_height))
    x = image.img_to_array(img)
   
    input_images.append(x) 
    input_images = np.array(input_images)

    """ 3. Make predictions """
    y_pred = model.predict(input_images)
    confidence_threshold = 0.5 # ПОРОГ ДОВЕРИЯ!!! САМ МОГУ УСТАНОВИТЬ!!!!

    y_pred_thresh = [y_pred[k][y_pred[k,:,1] > \
            confidence_threshold] for k in range(y_pred.shape[0])]
    np.set_printoptions(precision=2, suppress=True, linewidth=90)
    #print("Predicted boxes:\n")
    #print('   class   conf xmin   ymin   xmax   ymax')
    #print(y_pred_thresh[0])
    """ 4. For JSON-output """
    predictions = []
    for i in y_pred_thresh[0]:
        if fltr.count(classes[int(i[0])]):
            # Transform the predicted bounding boxes for the 300x300 image to the original image dimensions.
            xmin = int(i[2] * orig_images[0].shape[1] / img_width)  if i[2] > 0 else 0
            ymin = int(i[3] * orig_images[0].shape[0] / img_height) if i[3] > 0 else 0
            xmax = int(i[4] * orig_images[0].shape[1] / img_width)  if i[4] > 0 else 0
            ymax = int(i[5] * orig_images[0].shape[0] / img_height) if i[5] > 0 else 0

            cut_img = np.array(orig_images[0])[ymin:ymax, xmin:xmax]
            cut_img = image.array_to_img(cut_img) 
            """ Странные манипуляции для получения заголовка jpeg """
            cut_io = io.BytesIO()
            cut_img.save(cut_io, 'jpeg') # Взять формат который был указан на входном изображении и подставить сюда (jpeg, png и тд)
            cut_io.seek(0)
            crop_ = cut_io.read()
            cut_64_encode = img_header + base64.b64encode(crop_).decode()
            
            X, Y, height, width = xmin, ymin, ymax-ymin, xmax-xmin
            CLASS = classes[int(i[0])]
            confidence = str("%.1f%%" % (i[1]*100))
            predictions.append(({'class' : CLASS, 
                             'confidence' : confidence, 
                             'x' : X, 'y' : Y, 
                             'h' : height, 'w' : width,
                             'image' : cut_64_encode
                             }))  
    return predictions    
        
    
