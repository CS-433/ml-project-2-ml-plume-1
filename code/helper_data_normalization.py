'''Helper file for data normalization step.'''

#imports
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from helper_pca import *


def get_data_grayscale(file_path):
    '''Extract the data from the images in grayscale.'''
    data = [] # initialize data matrix
    filenames = os.listdir(file_path) # get filenames from path
    filenames_new = [] #initialize array that will contain the filenames that were not empty
    
    # loop through each file
    for file_idx, filename in enumerate(filenames):
        #extract image in grayscale
        img = cv2.imread(os.path.join(file_path,filename), cv2.IMREAD_GRAYSCALE)
        #skip to next image if current one is empty otherwise append to data
        if img is None: 
            pass
        else:
            data.append(img)
            filenames_new.append(filename)
            
    data = np.array(data)
    return filenames_new, data
    
    
def get_data_color(file_path):
    ''' Extract the data from the images in HSV.'''
    
    # intialize arrays
    h_vals = []
    s_vals = []
    v_vals = []
    filenames = os.listdir(file_path)
    
    # loop through each file
    for filename in filenames:
        img = cv2.imread(os.path.join(file_path,filename)) # read image in color
        if img is None:
            pass
        else:
            # if image is not empty separate into hsv values
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) 
            h,s,v = cv2.split(hsv)
            h_vals.append(h)
            s_vals.append(s)
            v_vals.append(v)
                            
    return h_vals, s_vals, v_vals


def normalize_brightness(file_path, method = 'histogram_flattening'):
    '''Normalize the images by a specified method.'''
    # get h,s,v values
    h_vals, s_vals, v_vals = get_data_color(file_path)
    results = [] # intialize results array
    
    # loop through each image and apply normalization method
    for v in v_vals:
        if method == 'histogram_flattening':
            result = cv2.equalizeHist(v)
            results.append(result)
        elif method == 'adaptive_histogram_flattening':
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(3, 3))
            result = clahe.apply(v)
            results.append(result)
        else:
            # data standardization
            s = 32
            m = 128
            v = (v-np.mean(v)) / np.std(v) * s + m
            result = np.array(v, dtype=np.uint8)
            results.append(result)

    data=[]
    # convert the results back into a grayscale image and append to data
    for result, h, s in zip(results, h_vals, s_vals):
        hsv = cv2.merge((h,s,result))
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        img_gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        data.append(img_gray)
        
    return data