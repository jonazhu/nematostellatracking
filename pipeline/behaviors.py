import pandas as pd
import numpy as np
import sys
import os
import yaml

import warnings

from tqdm import tqdm

def euclidean_distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def dist_bt_frames(x1, y1, x2, y2):
    #distance traveled by x and yin pixels
    x_dist_pixels = np.abs(x2 - x1)
    y_dist_pixels = np.abs(y2 - y1)

    #distance traveled: convert to microns
    y_dist_microns = y_dist_pixels * (dish_diameter_microns / dish_height_pixels)
    x_dist_microns = x_dist_pixels * (dish_diameter_microns / dish_width_pixels)

    #total distance in microns
    return np.sqrt(x_dist_microns * x_dist_microns + y_dist_microns * y_dist_microns)

def direction_bt_frames(x1, y1, x2, y2):
    a = (x2 - x1) * (dish_diameter_microns / dish_width_pixels)
    b = (y2 - y1) * (dish_diameter_microns / dish_height_pixels)

    #error handling for a being 0: if a is 0, we either go straight up or straight down
    if a == 0 and b < 0:
        return (np.pi / 2) 
    elif a == 0 and b > 0:
        return (1.5 * np.pi)
    elif a == 0 and b == 0:
        return np.nan #or we go nowhere

    angle = np.arctan(b / a)

    if a < 0: #left-hand side
        return angle + np.pi
    elif a > 0 and b < 0: #upper-right quadrant
        return angle + 2 * np.pi 
    else: #lower-right quadrant
        return angle
    
def get_speed_and_direction(df):
    dists = [np.nan]
    speeds = [np.nan]
    dirs = [np.nan]

    for i in range(len(df) - 1):
        x1 = df.iloc[i]["x_center"]
        y1 = df.iloc[i]["y_center"]
        x2 = df.iloc[i+1]["x_center"]
        y2 = df.iloc[i+1]["y_center"]

        next_dist = dist_bt_frames(x1, y1, x2, y2)
        next_speed = next_dist / time_between_frames
        next_dir = direction_bt_frames(x1, y1, x2, y2)

        if next_dist > tolerance:
            print("Warning: Distance of " + str(next_dist) + " microns traveled detected between images " + df.iloc[i]["image_filename"] + " and " + df.iloc[i+1]["image_filename"])

        dists.append(next_dist)
        speeds.append(next_speed)
        dirs.append(next_dir)

    df["distance"] = dists
    df["speed"] = speeds
    df["direction"] = dirs

    #not entirely necessary since we modify df itself
    return df


if __name__ == "__main__":

    print("Beginning behavior conversion process.")

    with open(sys.argv[1], "r") as fr:
        params = yaml.load(fr, yaml.Loader)

    if params["show_warnings"] == False:
        warnings.filterwarnings('ignore')
    
    os.chdir(params["image_dir"])

    df = pd.read_csv(params["tracked_labels"])

    dish_diameter_microns = params["dish_diameter"]
    dish_height_pixels = params["height_pixels"]
    dish_width_pixels = params["width_pixels"]
    time_between_frames = params["time_between_frames"]
    tolerance = params["distance_tolerance"] * (dish_diameter_microns / dish_height_pixels)

    #convert width, height to microns
    df["height"] = df["height"] * (dish_diameter_microns / dish_height_pixels)
    df["width"] = df["width"] * (dish_diameter_microns / dish_width_pixels)

    ids = df["id"].unique()
    df_individuals = []

    for i in ids:
        df_individuals.append(df[df.id == i])

    for d in tqdm(df_individuals):
        get_speed_and_direction(d)

    df_new = pd.concat(df_individuals)
    df_new.to_csv(params["behaviors"])
    print("Full behaviors saved to " + params["behaviors"])

    distance_changes = []
    for i in ids:
        distance_changes.append(euclidean_distance(
            df_individuals[i].iloc[0]["x_center"],
            df_individuals[i].iloc[0]["y_center"],
            df_individuals[i].iloc[-1]["x_center"],
            df_individuals[i].iloc[-1]["y_center"],
        ))

    df_summary = pd.concat([df_new[["id", "width", "height", "direction", "speed"]].groupby("id").mean(),
           df_new[["id", "distance"]].groupby("id").sum()], axis=1)
    df_summary.columns = ["mean_width_microns", "mean_height_microns", "avg_direction_radians", 
                        "avg_speed_mps", "total_distance_microns"]
    df_summary["distance_change"] = distance_changes
    df_summary.to_csv(params["summary"])
    
    print("Behavior summaries saved to " + params["summary"])
