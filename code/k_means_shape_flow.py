import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import adjusted_rand_score, silhouette_score

from matplotlib import pyplot as plt
# from sklearn.mixture import GaussianMixture
# from helper_clustering import import_data, visualize, obtain_clusters, info_clusters

# We import the data (already normalized over a year of samples)
data_path = r"C:\Users\Dan\Desktop\Hourly_Discharge.csv"
labels_path = r"C:\Users\Dan\Desktop\Labels_Clustering.csv"
data = pd.read_csv(data_path)
cols = data.columns
for col in cols:
    data[col] = data[col].astype(float)
data['Labels kmeans'] = -1
data['Labels gmm'] = -1
labels = pd.read_csv(labels_path)

# Merging datasets based on Unix Time
for index, row in labels.iterrows():
    name = labels.at[index,'Img name']
    unix_tmp = int(name[19:29])
    pos = round((unix_tmp-data.at[0,'Unix time'])/3600)
    data.at[pos,'Labels kmeans'] = labels.at[index,'Clusters kmeans']
    data.at[pos,'Labels gmm'] = labels.at[index,'Clusters gmm']

# Removing unnecessary lines in dataset
data = data.dropna()
for index, row in data.iterrows():
    if data.at[index,'Labels kmeans'] == -1 :
        data = data.drop([index])

# Separation of labels and data
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

# Comparison K-means/DBSCAN and with labeling K-means/GMM
ari_kmeans1 = adjusted_rand_score(labels_kmeans, kmeans.labels_).round(4)
ari_gmm1 = adjusted_rand_score(labels_gmm, kmeans.labels_).round(4)