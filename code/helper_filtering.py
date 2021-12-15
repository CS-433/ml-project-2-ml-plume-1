# imports
import cv2
import os  # to load the images from a folder
import numpy as np
from helper_clustering import import_data
from helper_edge_detection import edge_detection_img

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
        image (numpy array): the image whose number of pixels is going to be computed
    Returns:
        num_pixels (int): the number of pixels of the image given as input.
    '''
    num_pixels = image.shape[0]*image.shape[1]
    return num_pixels
    
def get_black_pixels(image):
    '''
    Compute the number of black pixels of an image.
    
    Args:
        image (numpy array): the image whose number of black pixels is going to be computed
    Returns:
        num_black_pixels (int): the number of black pixels of the image given as input.
    '''
    num_black_pixels = np.sum(image == 0)
    return num_black_pixels

def get_percentage(image):
    '''
    Compute the percentage of black pixels of an image.
    
    Args:
        image (numpy array): the image whose percentage of black pixels is going to be computed
    Returns:
        percentage (float): the percentage of black pixels of the image given as input.
    '''
    percentage = (get_black_pixels(image) / get_pixels(image)) * 100
    return percentage

def image_is_correct(image, tolerance, central):
    '''
    Determine is an image is correct (i.e. can be consider as a good entry).
    
    Args:
        image (numpy array): the image whose correcness is going to be determined
    Returns:
        correct (boolean): it will be true if the image is considered as a good input
                           for our classification problem.
    '''
    correct = True
    black_and_white = black_white(image)
    percentage = get_percentage(black_and_white)
    if (percentage <= 77 or percentage >= 89):
        correct = False
    return correct, percentage

def save_filtered_images(folder, folderSave):
    '''
    Function that is going to save the images contained in a folder that are good inputs
    for our classification problem into another folder.
    
    Args:
        folder (string): name of the folder from which the images are going to be load
                         to analize them
        folderSave (string): name of the folder in which the correct images are going
                         to be saved
    Returns:
        num_images (int): the number of images that have been saved in folderSave.
    '''
    num_images = 0
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename), cv2.IMREAD_GRAYSCALE)
        img_edges_roberts = edge_detection_img(img)
        if (image_is_correct(img_edges_roberts, tolerance, central)[0]):
            num_images += 1
            save_image(img, folderSave, filename)
    return num_images