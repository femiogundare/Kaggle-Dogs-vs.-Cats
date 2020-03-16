# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 10:51:23 2020

@author: femiogundare
"""

#import the necessary packages
import cv2

class MeanPreprocessor:
    def __init__(self, rMean, gMean, bMean):
        self.rMean = rMean
        self.gMean = gMean
        self.bMean = bMean
    def preprocess(self, image):
        #split the images into their respective RGB channels
        B, G, R = cv2.split(image.astype('float32'))
        
        #subtract the means from each channel
        R -= self.rMean
        G -= self.gMean
        B -= self.bMean
        
        #merge the channels back together, and return the preprocessed image
        return cv2.merge([B, G, R])