import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from matplotlib import pyplot as plt
# from sklearn.mixture import GaussianMixture
# from helper_clustering import import_data, visualize, obtain_clusters, info_clusters

# We import the data (already normalized over a year of samples)
# Input data and label path down here
# Features file contains these columns: Unix time, Flow rate, Temperature, Wind direction, Wind speed, Wind speed squared
# features in the Features file have already been normalized & standardized
# Labels files contains these columns: Img name
data_path = r"C:\Users\Dan\Desktop\EPFL\MA 5\Machine Learning\ml-project-2-ml-plume-1\data\Data_Part_2\Features_Part2.csv"
labels_path = r"C:\Users\Dan\Desktop\EPFL\MA 5\Machine Learning\ml-project-2-ml-plume-1\data\Data_Part_2\Labels_Clusters_Part2.csv"

data = pd.read_csv(data_path)
cols = data.columns
for col in cols:
    data[col] = data[col].astype(float)
labels = pd.read_csv(labels_path)
data['Labels kmeans'] = -1 
data['Labels gmm'] = -1


# Linking and merging both datasets based on Unix Time
for index, row in labels.iterrows():
    name = labels.at[index,'Image name']
    unix_tmp = int(name[19:29])
    pos = round((unix_tmp-data.at[0,'Unix time'])/3600)
    data.at[pos,'Labels kmeans'] = labels.at[index,'Clusters kmeans']
    data.at[pos,'Labels gmm'] = labels.at[index,'Clusters gmm']

# Removing unnecessary lines in dataset (No image or no features at that Unix time)
data = data.dropna()
for index, row in data.iterrows():
    if data.at[index,'Labels kmeans'] == -1 :
        data = data.drop([index])

# Separation of dataset into features, K-means labels and GMM labels
data = data.reset_index()
features = data[['Flow rate','Temperature','Wind direction','Wind speed','Wind speed squared']]
labels_kmeans = data[['Labels kmeans']]
labels_gmm = data[['Labels gmm']]
tmp1 = labels_kmeans.values
tmp2 = labels_gmm.values
labels_kmeans = tmp1.ravel()
labels_gmm = tmp2.ravel()

# We classify the data with k-means
np.random.seed(0)
kmeans = KMeans(n_clusters=4, init='random', n_init=100)
kmeans.fit(features)

# Comparison K-means and with labeling K-means/GMM
ari_kmeans1 = adjusted_rand_score(labels_kmeans, kmeans.labels_).round(4)
ari_gmm1 = adjusted_rand_score(labels_gmm, kmeans.labels_).round(4)

# We classify the data with k-means using only the flow rate
np.random.seed(0)
kmeans = KMeans(n_clusters=4, init='random', n_init=100)
features = data[['Flow rate']]
kmeans.fit(features)

# Comparison K-means and with labeling K-means/GMM
ari_kmeans1 = adjusted_rand_score(labels_kmeans, kmeans.labels_).round(4)
ari_gmm1 = adjusted_rand_score(labels_gmm, kmeans.labels_).round(4)