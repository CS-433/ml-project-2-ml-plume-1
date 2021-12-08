from os import listdir
from os.path import isdir, join
import pandas as pd

PATH_TO_FOLDERS = r"E:\Documents"
directories = [name for name in listdir(PATH_TO_FOLDERS) if isdir(name)]
directories = [name for name in directories if not name.startswith('.')]

folders_df = pd.DataFrame(directories)
