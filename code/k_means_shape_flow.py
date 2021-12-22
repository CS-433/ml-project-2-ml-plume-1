import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import adjusted_rand_score, silhouette_score

from matplotlib import pyplot as plt
# from sklearn.mixture import GaussianMixture
# from helper_clustering import import_data, visualize, obtain_clusters, info_clusters

# We import the data (already normalized over a year of samples)
pars_path = r"C:\Users\Dan\Desktop\Hourly_Discharge.csv"
labels_path = r"C:\Users\Dan\Desktop\Labels_Clustering.csv"
pars = pd.read_csv(pars_path)
cols = pars.columns
for col in cols:
    pars[col] = pars[col].astype(float)
data = pars
data['Clusters'] = 0
labels = pd.read_csv(labels_path)

# Merging datasets based on Unix Time
for index, row in labels.iterrows():
    name = labels.at[index,'Img name']
    unix_tmp = int(name[19:29])
    pos = round((unix_tmp-pars.at[0,'Unix time'])/3600)
    pars.at[pos,'Clusters'] = labels.at[index,'Cluster #']

# Removing unnecessary lines in dataset
data = data.dropna()
for index, row in data.iterrows():
    if data.at[index,'Clusters'] == 0 :
        data = data.drop([index])

# Separation of labels and data
data = data.reset_index()
features = data[['Flow rate','Temperature','Wind direction','Wind speed','Wind speed squared']]
true_labels = data[['Clusters']]
tmp = true_labels.values
true_labels = tmp.ravel()

# We classify the data with k-means
np.random.seed(1)
kmeans = KMeans(n_clusters=3, init='random', n_init=100)
kmeans.fit(features)

# We classify the data with DBSCAN
dbscan = DBSCAN(eps=0.3, min_samples=5)
dbscan.fit(features)

# Comparison with DBSCAN
kmeans_silhouette = silhouette_score(features, kmeans.labels_).round(2)
dbscan_silhouette = silhouette_score(features, dbscan.labels_).round(2)
ari_kmeans = adjusted_rand_score(true_labels, kmeans.labels_)
ari_dbscan = adjusted_rand_score(true_labels, dbscan.labels_)