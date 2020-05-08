import os
import shutil
import numpy as np
import pandas as pd

working_directory = os.path.dirname(os.getcwd())
dataset_dir = os.path.join(working_directory, "data_256")
scene_df = pd.read_csv(os.path.join(working_directory, "Scene hierarchy - Places365.csv"), index_col=0, header=[1])
outdoor_natural = scene_df.loc[scene_df['outdoor, natural']==True]

#drop first letter and single quotes from categories
categories = [x[4:-1] for x in outdoor_natural.index]

for root, dirs, files in os.walk(dataset_dir):
    folder_name = root.split("\\")[-1]
    if folder_name == "data_256":
        continue
    if folder_name not in categories and len(folder_name) > 1:
        print("Removing: " + folder_name)
        shutil.rmtree(root)
