# imports
import cv2
import os  # to load the images from a folder
import numpy as np
from helper_clustering import import_data
from helper_edge_detection import edge_detection_img

# useful functions
def normalize_image(image):
    '''
    Normalization of an image.
    
    Args:
        image (numpy array): image to normalize
    Returns:
        norm_image (numpy array): normalized image.
    '''
    # normalize the image
    img_grey_norm = np.zeros(image.shape)
    norm_image = cv2.normalize(image,img_grey_norm,0,255,cv2.NORM_MINMAX)
    return norm_image

def black_white(image):
    '''
    Conversion of an image into a binary image.
    
    Args:
        image (numpy array): image to convert into a black and white image
    Returns:
        img_binary (numpy array): binary version of the input image.
    '''
    # define a threshold
    final_thresh = np.mean(image)

    # threshold the image
    img_binary = cv2.threshold(image, final_thresh, 255, cv2.THRESH_BINARY)[1]
    
    return img_binary

def save_image(image, folder, filename):
    '''
    Save an image.
    
    Args:
        image (numpy array): image we want to save
        folder (string): name of the folder in which we want to save the image
        filename (string): future filename of the image we want to save
     '''
    # save image
    cv2.imwrite(os.path.join(folder,filename), image)

def get_pixels(image):
    '''
    Compute the number of pixels of an image.
    
    Args:
        image (numpy array): the image whose number of pixels is going to be compute
    Returns:
        num_pixels (int): the number of pixels of the image given as input.
    '''
    num_pixels = image.shape[0]*image.shape[1]
    return num_pixels
    
def get_black_pixels(image):
    '''
    Compute the number of black pixels of an image.
    
    Args:
        image (numpy array): the image whose number of black pixels is going to be compute
    Returns:
        num_black_pixels (int): the number of black pixels of the image given as input.
    '''
    num_black_pixels = np.sum(image == 0)
    return num_black_pixels

def get_percentage(image):
    '''
    '''
    return (get_black_pixels(image) / get_pixels(image)) * 100

def image_is_correct(image, tolerance, central):
    '''
    '''
    correct = True
    black_and_white = black_white(image)
    percentage = get_percentage(black_and_white)
    if (percentage <= 77 or percentage >= 89):
        correct = False
    return correct, percentage

def save_filtered_images(folder, folderSave):
    '''
    '''
    num_images = 0
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename), cv2.IMREAD_GRAYSCALE)
        img_edges_roberts = edge_detection_img(img)
        if (image_is_correct(img_edges_roberts, tolerance, central)[0]):
            num_images += 1
            save_image(img, folderSave, filename)
    return num_images