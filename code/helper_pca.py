import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def apply_pca(data, n_components, plot_explained_variance = True):
    '''Apply pca on feature matrix X with n_components desired components.''' 

    X = data.reshape(data.shape[0], data.shape[1]*data.shape[2]) # our feature matrix
    print(f'Before PCA, we have {X.shape[0]} samples, each with {X.shape[1]} features')
    
    pca = PCA(n_components) # initialize PCA
    pca.fit(X) # fit to data
    X_pca = pca.transform(X) # transform data
    imgs_reduced = pca.inverse_transform(X_pca) # get reduced images
    imgs_reduced = imgs_reduced.reshape(data.shape)
    
    print(f'After PCA, we have {X.shape[0]} samples, each with {X_pca.shape[1]} features')
    
    if plot_explained_variance:
        plt.grid()
        plt.plot(np.cumsum(pca.explained_variance_ratio_ * 100))
        plt.xlabel('Number of components')
        plt.ylabel('Explained variance')
        plt.tight_layout()
        
    return X_pca, imgs_reduced, pca


def visualize_img_reduction(data, imgs_reduced):
    # showing sample image
    img_grey_1 = data[0]
    img_grey_2 = data[1]
    img_reduced_1 = imgs_reduced[0]
    img_reduced_2 = imgs_reduced[1]

    fig, axs = plt.subplots(2,2, sharex=True, sharey=True, figsize = (12,12))
    axs[0,0].imshow(img_grey_1)
    axs[0,1].imshow(img_reduced_1)

    axs[1,0].imshow(img_grey_2)
    axs[1,1].imshow(img_reduced_2)

    axs[0,0].set_title('Image 1 Original')
    axs[0,1].set_title('Image 1 Reduced')
    axs[1,0].set_title('Image 2 Original')
    axs[1,1].set_title('Image 2 Reduced')

    plt.show()