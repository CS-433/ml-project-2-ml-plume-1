# imports
import cv2
import os  # to load the images from a folder
import numpy as np

reference_percentage = 48.26171875

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

def imageIsCorrect(image, tolerance, central):
    img_binary = convertBlackAndWhite(image)
    shape_1 = img_binary.shape[0]
    shape_2 = img_binary.shape[1]
    crop_image_binary = img_binary[int(shape_1/2):,int(shape_2/2):]
    percentage = getPercentage(crop_image_binary)
    correct = True
    if percentage < (reference_percentage - tolerance) or percentage > (reference_percentage + tolerance):
        correct = False
    else:
        crop_image_binary_2 = img_binary[int(shape_1/3):int((2*shape_1)/3),int(shape_2/3):int((2*shape_2)/3)]
        second_percentage = getPercentage(crop_image_binary_2)
        if second_percentage < central:
            correct = False
    return correct, percentage

def saveFilteredImages(folder, folderSave, tolerance=5, central=5):
    num_images = 0
    counter = 0
    # print("Reference Percentage " + str(reference_percentage))
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename), cv2.IMREAD_GRAYSCALE)
        # print("Percentage Image " + str(counter))
        counter += 1
        print(str(imageIsCorrect(img, tolerance, central)[1]))
        if (imageIsCorrect(img, tolerance, central)[0]):
            num_images += 1
            saveImage(normalizeImage(img), folderSave, filename)
    return num_images