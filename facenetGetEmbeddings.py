#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 15:59:19 2017

@author: KarimM
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import facenet

def getEmbeddings(image,image_size = 160):
  
    with tf.Graph().as_default():
      
        with tf.Session() as sess:
            
            # Read the file containing the pairs used for testing
            #pairs = lfw.read_pairs(os.path.expanduser(args.lfw_pairs))

            # Get the paths for the corresponding images
            paths = []
            paths.append(image)

            # Load the model
            facenet.load_model('/Users/KarimM/Desktop/project/medium-facenet-tutorial/etc/20170511-185253/20170511-185253.pb')
            
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            
            #image_size = images_placeholder.get_shape()[1]  # For some reason this doesn't work for frozen graphs
            embedding_size = embeddings.get_shape()[1]
        
            # Run forward pass to calculate embeddings
            print('Runnning forward pass on LFW images')
            nrof_batches = 1
            emb_array = np.zeros((1, embedding_size))
            for i in range(nrof_batches):
                images = facenet.load_data(paths, False, False, image_size)
                feed_dict = { images_placeholder:images, phase_train_placeholder:False }
                emb_array = sess.run(embeddings, feed_dict=feed_dict)
    print(emb_array)
    return emb_array

