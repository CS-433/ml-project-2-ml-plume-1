import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def apply_pca(data, n_components, plot_explained_variance = True):
    '''
    Apply pca on feature matrix X with n_components desired components.
    
    Args:
        data (numpy array): array with each image stored as its matrix representation
        n_components (int): number of desired final components after pca.
        plot_explained_variance (bool): if True, plots the explained variance vs. the number of components up to n_components.
        
    Returns:
        X_pca (numpy arrray): feature matrix after dimensionality reduction
        imgs_reduced (numpy array): recreated array with each image stored as its matrix representation created with just the pca features
    ''' 

    X = data.reshape(data.shape[0], data.shape[1]*data.shape[2]) # our feature matrix
    print(f'Before PCA, we have {X.shape[0]} samples, each with {X.shape[1]} features')
    
    pca = PCA(n_components) # initialize PCA
    pca.fit(X) # fit to data
    X_pca = pca.transform(X) # transform data
    imgs_reduced = pca.inverse_transform(X_pca) # get reduced images
    imgs_reduced = imgs_reduced.reshape(data.shape) # reshaped reduced images
    
    print(f'After PCA, we have {X.shape[0]} samples, each with {X_pca.shape[1]} features')
    
    #  plotting the explained variance to see how well the representation is at each value for number of components
    if plot_explained_variance:
        plt.grid()
        plt.plot(np.cumsum(pca.explained_variance_ratio_ * 100))
        plt.xlabel('Number of components')
        plt.ylabel('Explained variance')
        plt.tight_layout()
        
    return X_pca, imgs_reduced


def visualize_img_reduction(data, imgs_reduced, idxs):
    '''
    Plot what images look like before and after applying PCA to verify that important information is not lost.
    
    Args:
        data (numpy array): original image data before PCA
        imgs_reduced (numpy array): PCA reduced images transformed back into shape of original dat
        idxs (numpy array): indexes of images to be plotted
    '''
    num_imgs = len(idxs)
    num_cols = 2
    original_imgs = data[idxs]
    reduced_imgs = imgs_reduced[idxs]
    
    
    '''# showing sample image
    img_grey_1 = data[0]
    img_grey_2 = data[1]
    img_reduced_1 = imgs_reduced[0]
    img_reduced_2 = imgs_reduced[1]'''

    length = num_imgs*5
    width = num_cols*5
    plt.figure(figsize=(length,width)) # initialize figure
        
    for i in range(0, num_imgs):

        ax1 = plt.subplot(int(num_imgs), num_cols, (2*i)+1)
        ax1.imshow(original_imgs[i])
        ax1.set_title('Original')
        
        ax2 = plt.subplot(int(num_imgs), num_cols, (2*i)+2)
        ax2.imshow(reduced_imgs[i])
        ax2.set_title('Reduced')
        
        plt.axis('off')
        
    plt.show()
        