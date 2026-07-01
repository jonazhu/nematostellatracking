import ultralytics
from ultralytics import YOLO
import pandas as pd
import numpy as np

import os
import sys

#optional: a lot of warnings come up
import warnings
warnings.filterwarnings('ignore')

ultralytics.checks()

#TODO: integrate hyperparameter processing via YAML file

if __name__ == "__main__":
    best_model = sys.argv[1]
    images_dir = sys.argv[2]

    images_list = os.listdir(images_dir)
    images_list = [images_dir + "/" + i for i in images_list]

    df_full = pd.DataFrame() #the dataframe that contains everything from this stack

    for img in images_list:
        results = best_model(img, verbose=False)
        for r in results:
            b = r.boxes.data  # Boxes object for bounding box outputs
            df_res = pd.DataFrame(np.array(b))
            df_res.columns = ["x1", "y1", "x2", "y2", "conf", "class"]
            df_res.insert(0, "image_filename", img)
            df_full = pd.concat([df_full, df_res], ignore_index=True)

    df_full.to_csv(images_dir + "/positions.csv")
    print("Saved to " + images_dir + "/positions.csv")
