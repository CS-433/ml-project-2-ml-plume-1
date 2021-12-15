import numpy as np
from skimage import filters

def edge_detection(data):
    
    data_edges_roberts = []
    data_edges_sobel = []
    
    for image in data:
        data_edges_roberts.append(filters.roberts(image))
        data_edges_sobel.append(filters.sobel(image))
        
    data_edges_roberts = np.array(data_edges_roberts)
    data_edges_sobel = np.array(data_edges_sobel)
    
    return data_edges_roberts, data_edges_sobel

def edge_detection_img(img):
    img_edges_roberts = filters.roberts(img)
    return img_edges_roberts