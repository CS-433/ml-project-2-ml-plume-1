import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import matplotlib.image as mpimg
from helper_pca import *

def get_data_grayscale(file_path):
    
    data = []
    filenames = os.listdir(file_path)
    filenames_new = []
    
    for file_idx, filename in enumerate(filenames):
        img = cv2.imread(os.path.join(file_path,filename), cv2.IMREAD_GRAYSCALE)
        if img is None:
            pass
        else:
            data.append(img)
            filenames_new.append(filename)
            
    data = np.array(data)
    return filenames_new, data
    
    
def get_data_color(file_path):
    
    data = []
    h_vals = []
    s_vals = []
    v_vals = []
    filenames = os.listdir(file_path)
    filenames_new = []
    
    for filename in filenames:
        img = cv2.imread(os.path.join(file_path,filename)) 
        if img is None:
            pass
        else:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) 
            h,s,v = cv2.split(hsv)
            h_vals.append(h)
            s_vals.append(s)
            v_vals.append(v)
            data.append(img)
            filenames_new.append(filename)
            
    data = np.array(data)
                
    return filenames_new, data, h_vals, s_vals, v_vals


def normalize_brightness(filenames, path_to_out, h_vals, s_vals, v_vals, method = 'histogram_flattening'):
    
    results = []
    for v in v_vals:
        if method == 'histogram_flattening':
            result = cv2.equalizeHist(v)
            results.append(result)
        elif method == 'adaptive_histogram_flattening':
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(3, 3))
            result = clahe.apply(v)
            results.append(result)
        else:
            s = 32
            m = 128
            v = (v-np.mean(v)) / np.std(v) * s + m
            result = np.array(v, dtype=np.uint8)
            results.append(result)
    
    path_to_out = os.path.join(path_to_out, f'{method}/')
    for result, filename, h, s in zip(results, filenames, h_vals, s_vals):
        hsv = cv2.merge((h,s,result))
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        filename_out = os.path.join(path_to_out, f'result_{filename}')
        cv2.imwrite(filename_out, rgb) 