# Imports
import numpy as np
import csv

def write_labels(labels, filename):
    '''
    Write the labels in a file.
    
    Args:
        labels (numpy array): array containing the labels obtained as a result of applying a clustering algorithm
        filename: name of the file on which we are going to write the labels.
    '''
    f = open(filename, "w")
    for element in labels:
        f.write(str(element))
        f.write("\n")
    f.close()
    

def read_labels(filename, nlabels=3):
    '''
    Read the labels written with the previous function from a file.
    
    Args:
        filename: name of the file from which we are going to read the labels
        nlabels (int): the number of different labels that we have (which is the number of clusters that we have).
     
    Returns:
        labels (numpy arrray): array with the labels.
    '''
    file = open(filename)
    csvreader = csv.reader(file)
    
    rows = []
    for row in csvreader:
            rows.append(row)
            
    rows = np.resize(np.array(rows), len(rows))
    
    elems = []
    keys_list = [chr(i + ord('0')) for i in range(0, nlabels)]
    values_list = [i for i in range(0, nlabels)]
    
    # Create a map between the last two lists:
    zip_iterator = zip(keys_list, values_list)
    a_dictionary = dict(zip_iterator)
    
    for row in rows:
        elems.append(a_dictionary[row])
    return np.array(elems)