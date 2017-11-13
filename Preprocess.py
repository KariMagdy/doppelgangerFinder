#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 17:17:13 2017

@author: KarimM
"""

from __future__ import absolute_import
from __future__ import print_function
from scipy.misc import imread
import numpy as np
import csv
from scipy.misc import imresize 
import os.path

files = ['lfw-names.txt', 'pairsDevTest.txt', 'pairsDevTrain.txt']
def loadImage( basename, name, number):
    filename = "{0}/{1}/{1}_{2:04d}.jpg".format(basename, name, int(number))
    if os.path.isfile(filename):
        return imread(filename)
    return None

def loadImagePairFromRow(basename, r):
    if(len(r) == 3):
        # same
        im = np.array([loadImage( basename, r[0], r[1]), loadImage(basename, r[0], r[2])])
        if any(x is None for x in im):
            return None
        return im
    else:
        # different
        im = np.array([loadImage( basename, r[0], r[1]), loadImage( basename, r[2], r[3])])
        if any(x is None for x in im):
            return None
        return im

def loadLabelsFromRow(r):
    if(len(r) == 3):
        return 1
    else:
        return 0

# this should be equivalent to
#   np.array(map(lambda r:loadImagePairFromRow(tar, r), trainrows))
# but with a progress bar
def load_images(split, basename, rows):
    image_list = []
    labels = []
    for i, row in enumerate(rows):
        if (loadImagePairFromRow(basename, row) is None):
            continue
        else:
            image_list.append(loadImagePairFromRow( basename, row))
            labels.append(loadLabelsFromRow(row))
    return np.array(image_list), np.array(labels)

def convert_lfw(basename):
    tar_subdir = "lfw_funneled" if basename == "lfw-funneled" else basename

    print("--> Building test/train lists")
    # build lists, throwing away heading
    with open('pairsDevTrain.txt', 'rb') as csvfile:
        trainrows = list(csv.reader(csvfile, delimiter='\t'))[1:]
    with open('pairsDevTest.txt', 'rb') as csvfile:
        testrows = list(csv.reader(csvfile, delimiter='\t'))[1:]

    print("--> Converting")
    # extract all images in set
    train_images,train_labels = load_images("train", tar_subdir, trainrows)
    test_images,test_labels  = load_images("test",  tar_subdir, testrows)

    train_features = np.array([[f[0,:,:,0], f[0,:,:,1], f[0,:,:,2], f[1,:,:,0], f[1,:,:,1], f[1,:,:,2]] for f in train_images])
    test_features  = np.array([[f[0,:,:,0], f[0,:,:,1], f[0,:,:,2], f[1,:,:,0], f[1,:,:,1], f[1,:,:,2]] for f in test_images])

    train_targets = np.array([[n] for n in train_labels])
    test_targets  = np.array([[n] for n in test_labels])

    print("train shapes: ", train_features.shape, train_targets.shape)
    print("test shapes:  ", test_features.shape, test_targets.shape)

    return (train_features,train_targets,test_features,test_targets)


def cropImage(im):
    im2 = np.dstack(im).astype(np.uint8)
    # return centered 128x128 from original 250x250 (40% of area)
    newim = im2[61:189, 61:189]
    sized1 = imresize(newim[:,:,0:3], (32, 32), interp="bicubic", mode="RGB")
    sized2 = imresize(newim[:,:,3:6], (32, 32), interp="bicubic", mode="RGB")
    return np.asarray([sized1[:,:,0], sized1[:,:,1], sized1[:,:,2], sized2[:,:,0], sized2[:,:,1], sized2[:,:,2]])
