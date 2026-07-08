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
    with open(sys.argv[1], "r") as fr:
        params = yaml.load(fr, yaml.Loader)

    if params["run_yolo"] == False:
        print("Skipping YOLO model result extraction.")
    else:
        print("Beginning YOLO model result extraction.")

        if params["show_warnings"] == False:
            warnings.filterwarnings('ignore')
        os.chdir(params["image_dir"])

        best_model = YOLO(params["model"])

        images_list = os.listdir(params["image_dir"])
        #images_list = [params["image_dir"] + "/" + i for i in images_list]

        df_full = pd.DataFrame() #the dataframe that contains everything from this stack

        for img in tqdm(images_list):
            #check if we actually have an image
            if ".png" not in img:
                continue

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

        df_full.to_csv(params["labels"])
        print("Model results saved to " + params["labels"])
