# Imports
import os  # to load the images from a folder
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def import_data(directory, folder): # shouldnt we just put one path??
    '''Return numpy array containing all the data found in the given path.'''
    data = []
    os.chdir(directory)
    filenames = os.listdir(folder)
    
    # loop through each image file and add it to the data:
    for filename in filenames:
        img = cv2.imread(os.path.join(folder,filename), cv2.IMREAD_GRAYSCALE)
        data.append(img)
        
    return np.array(data)

def visualize(data, labels, fig_size, num_clusters=4, rows=128, columns=160):
    '''
    For each cluster, plot all the images contained in the cluster.
    
    Args:
        data (numpy array): feature matrix of the data 
        labels (numpy array): array with each index containing the cluster to which the image at that index belongs
        fig_size (int): size of the output figure in inches
        num_clusters (int): number of clusters in the data
        rows (int): number of rows in the matrix representation of each image 
        columns (int): number of columns in the matrix representation of each image 
    '''
    
    # Loop through each cluster to plot all the images in it
    for i in range(0,int(num_clusters)):

        cluster_idxs = np.where(labels==i)[0] # find the indexes of images in cluster i
        num_elements = cluster_idxs.shape[0]  # number of elements in the cluster
        num_rows = np.floor(num_elements/fig_size) # number of rows in the figure of the cluster

        print(f"cluster {i}, {num_elements} elements") 

        plt.figure(figsize=(fig_size,fig_size), dpi=300) # initialize figure
        
        # loop through each image in the cluster and plot it in a new subplot
        for k in range(0, num_elements):
            plt.subplot(int(num_rows+1), fig_size, k+1)
            image = data[cluster_idxs[k], ]
            image = image.reshape(rows, columns)
            plt.imshow(image, cmap='gray')
            plt.axis('off')
        plt.show()
        
        
def obtainClusters(folder, labels):
    '''
    Create a dictionary of clusters where each cluster label contains all the images belonging to it.
    
    Args:
        folder: path to folder containing all images
        labels (numpy array): array with each index containing the cluster to which the image at that index belongs
    Returns:
        groups (dict): dictionary with each cluster being a key and the values are the images in the cluster.
    '''
    filenames = os.listdir(folder) # retrieve files 
    groups = {} # initialize dict
    
    # loop through each file, cluster combination and add to dictionary.
    for file, cluster in zip(filenames, labels):
        if cluster not in groups.keys():
            groups[cluster] = []
            groups[cluster].append(file)
        else:
            groups[cluster].append(file)
            
    return groups


def infoClusters(labels, n):
    '''Print number of elements in each of n clusters with labels data.'''
    (unique, counts) = np.unique(labels, return_counts=True) 
    print("Number of clusters:", len(unique))
    for i in range(n):
        print("Cluster",i,"contains",counts[i],"elements.")
        

def create_labels_dict(num_clusters, labels):
    labels_dict = {}
    for cluster in range(num_clusters):
        cluster_ids = np.array([idx for idx, elem in enumerate(labels) if (labels[idx] == cluster)])
        labels_dict[cluster] = cluster_ids
    return labels_dict


def plot_cluster_examples(num_examples, num_clusters, labels, imgs_reduced):
    num_columns = 4
    num_rows = -(num_examples//-num_columns)

    for cluster in range(num_clusters):
        cluster_ids = np.array([idx for idx, elem in enumerate(labels) if (labels[idx] == cluster)])
        cluster_examples = np.random.choice(cluster_ids, size = num_examples, replace = False)

        fig, axs = plt.subplots(num_rows,num_columns, sharex=True, sharey=True, figsize = (5*num_columns, 4*num_rows))
        axs = axs.flatten()
        for ax, example in zip(axs, cluster_examples):
            ax.imshow(imgs_reduced[example],cmap='gray')
            ax.set_title(f'Image {example}')
        fig.suptitle(f'Cluster {cluster}')
        plt.tight_layout()