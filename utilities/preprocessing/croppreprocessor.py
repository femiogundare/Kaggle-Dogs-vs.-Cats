# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 12:02:32 2020

@author: femiogundare
"""


#import the required libraries
import numpy as np
import cv2

class CropPreprocessor:
    def __init__(self, width, height, horiz=True, inter=cv2.INTER_AREA):
        self.width = width
        self.height = height
        self.horiz = horiz
        self.inter = inter
        
    def preprocess(self, image):
        #initialize the lis of crops
        crops = []
        
        #grab the width and height of the image then use these
        #dimensions to define the corners of the image based
        (h, w) = image.shape[:2]
        coords = [
                [0, 0, self.width, self.height],
                [w - self.width, 0, w, self.height],
                [w - self.width, h - self.height, w, h],
                [0, h - self.height, self.width, h]]
        
        # compute the center crop of the image as well
        dW = int(0.5 * (w - self.width))
        dH = int(0.5 * (h - self.height))
        coords.append([dW, dH, w-dW, h-dH])
        
        #loop over the coordinates, extract each of the crops, and resize to a fixed size
        for startX, startY, endX, endY in coords:
            crop = image[startY:endY, startX:endX]
            crop = cv2.resize(crop, (self.height, self.width), interpolation=self.inter)
            crops.append(crop)
            
        #check to see if horizontal flip should be taken
        if self.horiz:
            #compute the horizontal mirror flips of each crop
            mirrors = [cv2.flip(c, 1) for c in crops]
            crops.extend(mirrors)
            
        #return the set of crops
        return np.array(crops)
        