#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 14:05:57 2018
Module for rest-like api of SSD
@author: kdg
"""
import base64
from flask import request, json 

from app import app
from app.SSD300 import detector

import tensorflow as tf
from keras import backend 
from keras.models import load_model

from app.keras_loss_function.keras_ssd_loss import SSDLoss
from app.keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from app.keras_layers.keras_layer_DecodeDetections import DecodeDetections
from app.keras_layers.keras_layer_L2Normalization import L2Normalization

backend.clear_session() # Clear previous models from memory.

_MODELS_PATH = 'app/SSD_300x300_ft_iter_120000.h5'
# We need to create an SSDLoss object in order to pass that to the model loader.
_SSD_LOSS = SSDLoss(neg_pos_ratio=3, n_neg_min=0, alpha=1.0)
_GRAPH = tf.get_default_graph()
_MODEL = load_model(_MODELS_PATH, compile = False,
                   custom_objects = {'AnchorBoxes': AnchorBoxes,
                                     'L2Normalization': L2Normalization,
                                     'DecodeDetections': DecodeDetections,
                                     'compute_loss': _SSD_LOSS.compute_loss})

@app.route('/ssd', methods=['POST'])
def ssd():
    
    if (not is_valid_request(request)):
            return json.jsonify(get_json_response(msg='Invalid request'))

    img_b64, _ = get_request_data(request)
    img_body = get_image_body(img_b64)
    img_header = get_image_header(img_b64)
    
    if (img_body is None):
        return json.jsonify(get_json_response(msg='Image not found'))

    #start_time = datetime.now()
    with _GRAPH.as_default():
        prediction_result = detector(img_body, img_header, _MODEL) 
    #delta = datetime.now() - start_time

    if (prediction_result == []):
        return json.jsonify(get_json_response())
    # print(delta.total_seconds() * 1000.0)
    return json.jsonify(get_json_response(prediction_result))

def is_valid_request(request):
    return 'image' in request.json

def get_request_data(request):
    r = request.json
    image = r['image'] if 'image' in r else ''
    min_accuracy = r['minAccuracy'] if 'minAccuracy' in r else 60
    return image, min_accuracy

def get_image_body(img_b64):
    if 'data:image' in img_b64:
        img_encoded = img_b64.split(',')[1]
        return base64.decodebytes(img_encoded.encode('utf-8'))
    else:
        return None
    
def get_image_header(img_b64):
    if 'data:image' in img_b64:
        #data:image/jpeg;base64,
        return img_b64.split(',')[0] + ','
    else:
        return None     

def get_json_response(result=None, msg=None):
    json = {
        'success': False
    }

    if msg is not None:
        json['message'] = msg
        return json

    json['data'] = []

    if result is None:
        return json

    for item in result:
        json['data'].append(item)

    json['success'] = True
    return json

if __name__ == '__main__':
    app.run(port='8080', debug=True)