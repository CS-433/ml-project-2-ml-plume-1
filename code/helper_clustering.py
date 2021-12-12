# Imports
import numpy as np
import matplotlib.pyplot as plt
import os  # to load the images from a folder
import cv2
from sklearn.decomposition import PCA

def importData(directory, folder):
    data = []
    os.chdir(directory)
    filenames = os.listdir(folder)
    for filename in filenames:
            img = cv2.imread(os.path.join(folder,filename), cv2.IMREAD_GRAYSCALE)
            data.append(img)
    return np.array(data)

def visualize(data, Z, size, n=4, rows=160, columns=128):
    '''
        n: number of clusters
    '''
    # We plot the resulting clusters
    for i in range(0,int(n)):

        row = np.where(Z==i)[0] # row in Z for elements of cluster i
        num = row.shape[0]      # number of elements for each cluster
        r = np.floor(num/20.)   # number of rows in the figure of the cluster

        print("cluster " + str(i))
        print(str(num) + " elements")

        plt.figure(figsize=(size,size), dpi=300)
        for k in range(0, num):
            plt.subplot(int(r+1), size, k+1)
            image = data[row[k], ]
            image = image.reshape(rows, columns)
            plt.imshow(image, cmap='gray')
            plt.axis('off')
        plt.show()
        
def obtainClusters(folder, labels):
    filenames = os.listdir(folder)
    groups = {}
    for file, cluster in zip(filenames, labels):
        if cluster not in groups.keys():
            groups[cluster] = []
            groups[cluster].append(file)
        else:
            groups[cluster].append(file)
    return groups

def infoClusters(labels, n):
    (unique, counts) = np.unique(labels, return_counts=True)
    print("Number of clusters:", len(unique))
    for i in range(n):
        print("Cluster",i,"contains",counts[i],"elements.")

# helpful functions for PCA

def apply_pca(data, n_components):
    '''Apply pca on feature matrix X with n_components desired components.''' 

    X = data.reshape(data.shape[0], data.shape[1]*data.shape[2]) # our feature matrix
    print(f'Before PCA, we have {X.shape[0]} samples, each with {X.shape[1]} features')
    
    pca = PCA(n_components) # initialize PCA
    pca.fit(X) # fit to data
    X_pca = pca.transform(X) # transform data
    imgs_reduced = pca.inverse_transform(X_pca) # get reduced images
    imgs_reduced = imgs_reduced.reshape(data.shape)
    
    print(f'After PCA, we have {X.shape[0]} samples, each with {X_pca.shape[1]} features')
    return X_pca, imgs_reduced, pca