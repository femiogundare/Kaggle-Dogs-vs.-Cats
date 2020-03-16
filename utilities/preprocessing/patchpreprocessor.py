# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 11:48:27 2020

@author: femiogundare
"""


#import the required libraries
from sklearn.feature_extraction.image import extract_patches_2d


class PatchPreprocessor:
    def __init__(self, height, width):
        #store the target height and width of the image
        self.height = height
        self.width = width
    def preprocess(self, image):
        #extract a random crop from the image with the target height and width
        return extract_patches_2d(
                image, patch_size=(self.height, self.width), max_patches=1)[0]