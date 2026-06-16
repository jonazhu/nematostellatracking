import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("tracking_test_v2.csv")
files = df["image_filename"].unique()
df_individuals = []
for f in files:
    df_individuals.append(df[df.image_filename == f])

def get_frame(idx):
    img_current = plt.imread(df_individuals[idx].iloc[0]["image_filename"])
    fig, ax = plt.subplots()
    ax.imshow(img_current, cmap="gray")
    for i in range(len(df_individuals[idx])):
        w = df_individuals[idx].iloc[i]["x2"] - df_individuals[idx].iloc[i]["x1"]
        h = df_individuals[idx].iloc[i]["y2"] - df_individuals[idx].iloc[i]["y1"]
        x = df_individuals[idx].iloc[i]["x2"] - w
        y = df_individuals[idx].iloc[i]["y2"] - h
        if df_individuals[idx].iloc[i]["class"] == 0.0:
            rect = plt.Rectangle((x, y), w, h, edgecolor='r', facecolor='none')
        else: 
            rect = plt.Rectangle((x, y), w, h, edgecolor='b', facecolor='none')
        ax.add_patch(rect)

    return fig