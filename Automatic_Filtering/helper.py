# imports
import cv2
import os  # to load the images from a folder
import numpy as np

# useful functions
def convertBlackAndWhite(image):
    # normalize the image
    img_grey_norm = np.zeros(image.shape)
    
    # define a threshold
    final_thresh = np.mean(image)

    # threshold the image
    img_binary = cv2.threshold(image, final_thresh, 255, cv2.THRESH_BINARY)[1]
    
    return img_binary

def saveImage(image, folder, filename):
    # save image
    cv2.imwrite(os.path.join(folder,filename), image)

def getNumPixels(image):
    return np.sum(image >= 0)
    
def getNumBlackPixels(image):
    return np.sum(image == 0)

def getPercentage(image):
    return (getNumBlackPixels(image)/getNumPixels(image))*100

def imageIsCorrect(image):
    percentage = getPercentage(image)
    crop_image = image[256:,:]
    percentage_2 = getPercentage(crop_image)
    plt.imshow(crop_image)
    plt.show()
    correct = True
    if percentage < 54:
        if percentage_2 < 9 or percentage_2 > 37:
            correct = False
            print("Incorrect percentage_2: " + str(percentage_2))
        else:
            print("Correct percentage_2: " + str(percentage_2))
            
    return correct