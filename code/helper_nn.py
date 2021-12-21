import numpy as np
import csv

def write_labels(labels, filename):
    f = open(filename, "w")
    for element in labels:
        f.write(str(element))
        f.write("\n")
    f.close()
    

def read_labels(filename, nlabels=3):
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
        #if row == '0':
        #    elems.append(0)
        #elif row == '1':
        #    elems.append(1)
        #else:
        #    elems.append(2)
        elems.append(a_dictionary[row])
    return np.array(elems)