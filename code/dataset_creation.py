import os
from PIL import Image
import time
import random
import shutil

# Path of the picture folder (old_dir) and of the dataset folder (new_dir)
new_dir = r"D:\ML_Data\Full_Data"
old_dir = r"E:\TLB1_thermal - Copy\images"
Subfs=[]
Picts_old=[]
Picts_new=[]

start_time = time.time()

## Gathers every picture path into Picts_old
for root, subdirectories, files in os.walk(old_dir):
        for subdirectory in subdirectories:
            subf = os.path.join(root, subdirectory)
            Subfs.append(subf)

        for file in files:
            pict = os.path.join(root, file)
            Picts_old.append(pict)
    
print("--- %s seconds ---" % (time.time() - start_time))
start_time = time.time()

# For every picture, renames it with date (year-month-day) and adds it in another folder with lower res (16x less px)
for i in Picts_old:
    old_img = i
    if len(old_img) != 67:
        old_img = old_img[0:40] + old_img[47:74]
    new_img_name = 'thermal_' + old_img[30:40] + old_img[48:]
    new_img = os.path.join(new_dir, new_img_name)
    img = Image.open(old_img)
    img = img.resize((160,128),Image.ANTIALIAS)
    img.save(new_img)
    Picts_new.append(new_img)
        
print("--- %s seconds ---" % (time.time() - start_time))
start_time = time.time()

# Creation of dataset 1: With 3'000 pictures divided in 3 sets
Picts_rand = random.sample(Picts_new, 3000)

Picts_d1_train = Picts_rand[0:1000]
for i in Picts_d1_train:
    new_path = os.path.join(r"D:\ML_Data\Dataset_3k\Training_set", i[21:])
    shutil.copyfile(i, new_path)
    
Picts_d1_valid = Picts_rand[1000:2000]
for i in Picts_d1_valid:
    new_path = os.path.join(r"D:\ML_Data\Dataset_3k\Validation_set", i[21:])
    shutil.copyfile(i, new_path)
    
Picts_d1_test = Picts_rand[2000:3000]
for i in Picts_d1_test:
    new_path = os.path.join(r"D:\ML_Data\Dataset_3k\Testing_set", i[21:])
    shutil.copyfile(i, new_path)
    
# Creation of dataset 2: With 9'000 pictures divided in 3 sets
Picts_rand = random.sample(Picts_new, 15000)

Picts_d2_train = Picts_rand[0:5000]
for i in Picts_d2_train:
    new_path = os.path.join(r"D:\ML_Data\Dataset_15k\Training_set", i[21:])
    shutil.copyfile(i, new_path)
    
Picts_d2_valid = Picts_rand[5000:10000]
for i in Picts_d2_valid:
    new_path = os.path.join(r"D:\ML_Data\Dataset_15k\Validation_set", i[21:])
    shutil.copyfile(i, new_path)
    
Picts_d2_test = Picts_rand[10000:15000]
for i in Picts_d2_test:
    new_path = os.path.join(r"D:\ML_Data\Dataset_15k\Testing_set", i[21:])
    shutil.copyfile(i, new_path)
    
# Creation of dataset 3: With 30'000 pictures divided in 3 sets
Picts_rand = random.sample(Picts_new, 75000)

Picts_d3_train = Picts_rand[0:25000]
for i in Picts_d3_train:
    new_path = os.path.join(r"D:\ML_Data\Dataset_75k\Training_set", i[21:])
    shutil.copyfile(i, new_path)

Picts_d3_valid = Picts_rand[25000:50000]
for i in Picts_d3_valid:
    new_path = os.path.join(r"D:\ML_Data\Dataset_75k\Validation_set", i[21:])
    shutil.copyfile(i, new_path)
    
Picts_d3_test = Picts_rand[50000:75000]
for i in Picts_d3_test:
    new_path = os.path.join(r"D:\ML_Data\Dataset_75k\Testing_set", i[21:])
    shutil.copyfile(i, new_path)

print("--- %s seconds ---" % (time.time() - start_time))
