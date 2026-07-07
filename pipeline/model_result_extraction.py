import ultralytics
from ultralytics import YOLO
import pandas as pd
import numpy as np
import os
import sys
import yaml

import warnings

from tqdm import tqdm

ultralytics.checks()

#TODO: integrate hyperparameter processing via YAML file
# integrate preprocessing of YOLO results

if __name__ == "__main__":

    print("Beginning YOLO model result extraction.")

    with open(sys.argv[1], "r") as fr:
        params = yaml.load(fr, yaml.Loader)

    if params["show_warnings"] == False:
        warnings.filterwarnings('ignore')
    
    os.chdir(params["image_dir"])

    best_model = sys.argv[1]
    images_dir = sys.argv[2]

    images_list = os.listdir(images_dir)
    images_list = [images_dir + "/" + i for i in images_list]

    df_full = pd.DataFrame() #the dataframe that contains everything from this stack

    for img in tqdm(images_list):
        results = best_model(img, verbose=False)
        for r in results:
            b = r.boxes.data  # Boxes object for bounding box outputs
            df_res = pd.DataFrame(np.array(b))
            df_res.columns = ["x1", "y1", "x2", "y2", "conf", "class"]
            df_res.insert(0, "image_filename", img)
            df_full = pd.concat([df_full, df_res], ignore_index=True)

    #section is run if you want to preprocess the labels; that is, removing boxes with IoU
    #greater than the given parameter, and checking distances between things
    if params["preprocess"] == True:
        print("Preprocessing labels from " + params["labels"])

    df_full.to_csv(images_dir + "/positions.csv")
    print("Saved to " + images_dir + "/positions.csv")
