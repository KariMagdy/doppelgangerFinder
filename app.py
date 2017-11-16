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
    file.save(f)
    embeddings1 = getEmbeddings(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
    file = request.files['imageII']
    fII = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(fII)
    embeddings2 = getEmbeddings(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff),1)
    samescore = str(float(4.0 - dist)/4.0 * 100.0)
    #samescore = str(float((4.0 - np.linalg.norm(embeddings1-embeddings2))/4.0 * 100.0))
    return render_template('index.html', similarity=samescore, init=True)