#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 10:49:02 2018
SSD300 Inference by trained model
@author: kdg
"""

from keras import backend as K
from keras.preprocessing import image
from keras.optimizers import Adam
import numpy as np
import cv2
import base64
import io

from app.models.keras_ssd300 import ssd_300
from app.keras_loss_function.keras_ssd_loss import SSDLoss

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

def detector(image_body, img_header): #image_format):
    """ 1. Load a trained SSD
    1.2. Load a trained model """
    # 1: Build the Keras model
    K.clear_session() # Clear previous models from memory.
    model = ssd_300(image_size=(img_height, img_width, 3),
                n_classes=20,
                mode='inference',
                l2_regularization=0.0005,
                scales=[0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05], # The scales for MS COCO are [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05]
                aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5],
                                         [1.0, 2.0, 0.5]],
                two_boxes_for_ar1=True,
                steps=[8, 16, 32, 64, 100, 300],
                offsets=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                clip_boxes=False,
                variances=[0.1, 0.1, 0.2, 0.2],
                normalize_coords=True,
                subtract_mean=[123, 117, 104],
                swap_channels=[2, 1, 0],
                confidence_thresh=0.5,
                iou_threshold=0.45,
                top_k=200,
                nms_max_output_size=400)    
    # 2: Load the trained weights into the model.
    # TODO: Set the path of the trained weights.
    weights_path = 'app/VGG_VOC0712_SSD_300x300_ft_iter_120000.h5'
    model.load_weights(weights_path, by_name=True)

    # 3: Compile the model so that Keras won't complain the next time you load it.
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
    model.compile(optimizer=adam, loss=ssd_loss.compute_loss)    
    
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
        
    
