import pandas as pd
import numpy as np
import os
import sys
import yaml

from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

#TODO: 
#modify tracking preprocessing of positions (details written on paper - will experiment in notebook first)
#Integrate further hyperparameter processing

#helper functions for making a cost matrix from two dataframes
def euclidean_distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def cost_matrix(df_first, df_second):
    cm = np.zeros((len(df_first), len(df_second)))
    for i in range(len(df_first)):
        for j in range(len(df_second)):
            x1 = df_first.iloc[i]["x_center"]
            y1 = df_first.iloc[i]["y_center"]
            x2 = df_second.iloc[j]["x_center"]
            y2 = df_second.iloc[j]["y_center"]

            cm[i, j] = euclidean_distance(x1, y1, x2, y2)
    
    return cm

#helper function: take the ids of one dataframe, then assign corresponding ones to the second dataframe
#based on row and column indices (all given as parameters)
def assign_ids(df_first, df_second, row_ind, col_ind):
    new_ids = np.full(len(df_second), -1)

    #this routine handles if df_first has the same number or fewer entries as df_second. 
    #In this case, any extra labels are simply discarded at the end of assignment.
    #Do note that this can still lead to ways in which things can go wrong.
    if len(df_first) <= len(df_second):
        #go through the IDs in the first df
        for i in range(len(df_first)):
            current_id = df_first.iloc[i]["id"]
            new_index = col_ind[i]
            new_ids[new_index] = current_id

        df_second["id"] = new_ids

    #case in which df_first has more entries than df_second.
    #this will attempt to 
    elif len(df_first) > len(df_second):
        ids_first = [id for id in range(len(df_first))]
        used_ids = []
        #same iteration as above but with some slight changes
        for i in range(len(df_second)):
            current_id = df_first.iloc[i]["id"]
            used_ids.append(current_id)
            new_index = col_ind[i]
            new_ids[new_index] = current_id

        df_second["id"] = new_ids

        unused_ids = [id for id in ids_first if id not in used_ids]
        for x in unused_ids:
            pd.concat([df_second, df_first[df_first.id == x]], ignore_index=True)
            
        df_second["image_filename"] = df_second.iloc[0]["image_filename"]

    else:
        print("Something went wrong in assigning IDs")
        assert False

if __name__ == "__main__":
    print("Beginning tracking process.")

    with open(sys.argv[1], "r") as fr:
        params = yaml.load(fr, yaml.Loader)
    
    os.chdir(params["image_dir"])

    df = pd.read_csv(params["labels"])

    #section is run if annotations.csv is from the web interface, where the dataframe has a yolo_class
    if "yolo_class" in df.columns:
        print("YOLO assisted columns detected. Renaming some columns and items.")
        df = df.filter(items = ["filename", "class", "x1", "y1", "x2", "y2"])
        df.columns = ["image_filename", "class", "x1", "y1", "x2", "y2"] #rename for consistency
        classes = []
        for i in tqdm(range(len(df))):
            if "Polyp" in df.iloc[i]["class"]:
                classes.append("Polyp")
            elif "Planula" in df.iloc[i]["class"]:
                classes.append("Planula")
            else:
                print("Warning: Image " + df.iloc[i]["image_filename"] + " has an improperly named label")

        df["class"] = classes


    #adding attributes to dataframe based on positions
    df["width"] = df["x2"] - df["x1"]
    df["height"] = df["y2"] - df["y1"]
    df["x_center"] = df["x2"] - 0.5 * df["width"]
    df["y_center"] = df["y2"] - 0.5 * df["height"]

    filenames = df["image_filename"].unique()
    filenames = np.sort(filenames) #sorting only the filenames makes this a lot quicker later
    df_individuals = []

    for f in filenames:
        df_individuals.append(df[df.image_filename == f])

    #automatically assigns IDs for the first dataframe
    df_individuals[0]["id"] = np.arange(len(df_individuals[0]))

    print("Initial calculations completed. Computing actual tracking (this may take a while...)")

    df_lens = [len(d) for d in df_individuals]

    for i in tqdm(range(len(df_individuals) - 1)):
        if df_lens[i] < df_lens[i+1]:
            print("Warning: " + df_individuals[i].iloc[0]["image_filename"] + 
                  " has fewer labels than " + df_individuals[i+1].iloc[0]["image_filename"])
            
        if df_lens[i] > df_lens[i+1]:
            print("Warning: " + df_individuals[i].iloc[0]["image_filename"] + 
                  " has more labels than " + df_individuals[i+1].iloc[0]["image_filename"])

        cm = cost_matrix(df_individuals[i], df_individuals[i+1])
        row_ind, col_ind = linear_sum_assignment(cm)

        assign_ids(df_individuals[i], df_individuals[i+1], row_ind, col_ind)
        df_individuals[i+1] = df_individuals[i+1][df_individuals[i+1].id != -1]

    df_full = pd.concat(df_individuals)
    df_full.to_csv(params["tracked_labels"], index = False)
    print("Tracked annotations saved to " + params["tracked_labels"])