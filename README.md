# ml-project-2-ml-plume-1

## Table of Contents
1. [General Info](#general-info)
2. [Directory & File Structure](#directory-and-file-structure)
3. [How to run our project](#how-to-run-our-project)
5. [Members of the team](#members-of-the-team)
### General Information
***
River plumes are particularly important to understand marine and aquatic coastal environments. The aim of this project is to understand and predict Rhône's plume shape.

The data set is original data from the lab [ECOL](https://www.epfl.ch/labs/ecol/) from the EPFL.

### Directory and File Structure
***
```
project
│   README.md
|   Report.pdf
│
└───code
|     └───helpers
|       └───Dataset_creation.py
|       └───helper_clustering.py
|       └───helper_normalization.py
|       └───helper_edge_detection.py
|       └───helper_filtering.py
|       └───helper_nn.py
|       └───helper_pca.py
|       └───k_means_shape_flow.py
|     automatic_generation_of_filtered_data.ipynb
|     convolutional_nn.ipynb
|     image_classification.ipynb
|     labels_data.csv
|     processing_clustering.ipynb
|
└───data
|     └───Cluster_Examples
|           └───cluster1_bad_images
|           └───cluster2_triangle_with_overflow
|           └───cluster3_triangle_without_overflow
|           └───cluster4_patatoid_with_overflow
|     └───Data_Part_2
|           Features_Part2.csv
|           Labels_Clusters_Part2.csv
|     └───Save_3K
|     └───Save_15K
|     images.zip
|     training_labels.csv
|
```

The folder ```Cluster_Examples``` contains examples of the images provided by the lab that we are going to use as the input data for developping our project.

The folders ```Save_3K``` and ```Save_15K``` are the images obtained after filtering the bad images from the training data set of the 3K data set and the training data set of the 15K data set.

The file ```images.zip``` will be used in the ```convolutional_nn.ipynb``` file (it is just used for uploading the images easily to *Google Colab*).

## How to run our project
***
In these section we will make some notes on how particular parts of our project should be run.

We run the convolutional_nn.ipynb file in *Google Colab*. Note that we need to upload the following files to properly run it: ```images.zip```, ```helper_clustering.py``` and ```training_labels.csv```.

## Members of the team
***
* Paula Dolores Rescala
* María Isabel Ruiz Martínez
* Gönczy Daniel Alessandro Laszlo
