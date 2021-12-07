# imports
import cv2
import os  # to load the images from a folder
import numpy as np

# useful functions
def normalizeImage(image):
    # normalize the image
    img_grey_norm = np.zeros(image.shape)
    norm_image = cv2.normalize(image,img_grey_norm,0,255,cv2.NORM_MINMAX)
    return norm_image

def convertBlackAndWhite(image):
    # define a threshold
    final_thresh = np.mean(image)

    # threshold the image
    img_binary = cv2.threshold(image, final_thresh, 255, cv2.THRESH_BINARY)[1]
    
    return img_binary

def saveImage(image, folder, filename):
    # save image
    cv2.imwrite(os.path.join(folder,filename), image)

def getNumPixels(image):
    return image.shape[0]*image.shape[1]
    
def getNumBlackPixels(image):
    return np.sum(image == 0)

def getPercentage(image):
    return (getNumBlackPixels(image)/getNumPixels(image))*100

def imageIsCorrect(image):
    percentage = getPercentage(image)
    crop_image = image[256:,:]
    percentage_2 = getPercentage(crop_image)
    correct = True
    if percentage < 54:
        if percentage_2 < 9 or percentage_2 > 37:
            correct = False       
    return correct

def saveFilteredImages(folder, folderSave, crop=False):
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename), cv2.IMREAD_GRAYSCALE)
        # if crop:
        #    img = img[190:,190:] # we crop the image
        img_binary = convertBlackAndWhite(img)
        if (imageIsCorrect(img_binary)):
            saveImage(normalizeImage(img), folderSave, filename)
          