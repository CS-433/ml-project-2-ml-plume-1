'''Helper file for edge detection.'''
# imports
import numpy as np
from skimage import filters

def edge_detection(data):
    '''Return the data after edge detection with both roberts and sobel operators.'''
    # initialize new data
    data_edges_roberts = []
    data_edges_sobel = []
    
    # loop through each image
    for image in data:
        # apply filters
        data_edges_roberts.append(filters.roberts(image))
        data_edges_sobel.append(filters.sobel(image))
        
    # convert to numpy array
    data_edges_roberts = np.array(data_edges_roberts)
    data_edges_sobel = np.array(data_edges_sobel)
    
    return data_edges_roberts, data_edges_sobel

def edge_detection_img(img):
    '''Return data after roberts operator applied for edge detection.'''
    img_edges_roberts = filters.roberts(img)
    return img_edges_roberts