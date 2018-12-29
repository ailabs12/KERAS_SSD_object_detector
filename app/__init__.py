#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 14:03:39 2018

@author: kdg
"""

from flask import Flask

app = Flask(__name__)

from app import routes