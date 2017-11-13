#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 20:13:23 2017

@author: KarimM
"""

import os
from flask import Flask, render_template, request
from facenetGetEmbeddings import getEmbeddings
import numpy as np 

app = Flask(__name__)

UPLOAD_FOLDER = os.path.basename('uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def hello_world():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['image']
    f = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    # add your custom code to check that the uploaded file is a valid image and not a malicious file (out-of-scope for this post)
    y = getEmbeddings(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
    z = getEmbeddings('/Users/KarimM/Desktop/data/minecleaned/faceIV/faceIV.png')
    samescore = str(float(np.dot(y,np.transpose(z))))
    file.save(f)
    return render_template('index.html', similarity=samescore, init=True)