import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import fastgif

import warnings
warnings.filterwarnings('ignore')

#TODO:
#clean general structure
#integrate hyperparameter processing
#add one more type of animated GIF that overlays paths as they are being made - will experiment in notebook first

try:
    df = pd.read_csv(sys.argv[1])
except:
    df = pd.read_csv("behaviors_full.csv")
files = df["image_filename"].unique()
df_files = []
for f in files:
    df_files.append(df[df.image_filename == f])

def get_frame(idx):
    img_current = plt.imread(df_individuals[idx].iloc[0]["image_filename"])
    fig, ax = plt.subplots()
    ax.imshow(img_current, cmap="gray")
    for i in range(len(df_files[idx])):
        w = df_files[idx].iloc[i]["x2"] - df_files[idx].iloc[i]["x1"]
        h = df_files[idx].iloc[i]["y2"] - df_files[idx].iloc[i]["y1"]
        x = df_files[idx].iloc[i]["x2"] - w
        y = df_files[idx].iloc[i]["y2"] - h
        if df_files[idx].iloc[i]["class"] == 0.0:
            rect = plt.Rectangle((x, y), w, h, edgecolor='r', facecolor='none')
        else: 
            rect = plt.Rectangle((x, y), w, h, edgecolor='b', facecolor='none')
        ax.add_patch(rect)

    return fig

def get_direction_numbers(df):
    radii = np.zeros(8)
    for i in range(len(df)):
        d = df.iloc[i]["direction"]
        if d < (np.pi / 8) or d >= (15 * np.pi / 8):
            radii[0] += 1
        elif d < (3 * np.pi / 8) and d >= (np.pi / 8):
            radii[1] += 1
        elif d < (5 * np.pi / 8) and d >= (3 * np.pi / 8):
            radii[2] += 1
        elif d < (7 * np.pi / 8) and d >= (5 * np.pi / 8):
            radii[3] += 1
        elif d < (9 * np.pi / 8) and d >= (7 * np.pi / 8):
            radii[4] += 1
        elif d < (11 * np.pi / 8) and d >= (9 * np.pi / 8):
            radii[5] += 1
        elif d < (13 * np.pi / 8) and d >= (11 * np.pi / 8):
            radii[6] += 1
        elif d < (15 * np.pi / 8) and d >= (13 * np.pi / 8):
            radii[7] += 1

    return radii

if __name__ == "__main__":
    try:
        df_summary = pd.read_csv(sys.argv[2])
    except:
        df_summary = pd.read_csv("behaviors_summary.csv")

    ids = df["id"].unique()
    df_individuals = []

    for i in ids:
        df_individuals.append(df[df.id == i])

    for d in df_individuals:
        N = 8
        theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
        radii = get_direction_numbers(d)
        width = np.pi / 4 * np.ones(8)

        colors = plt.colormaps["viridis"](radii / 10.)

        ax = plt.subplot(projection='polar')
        ax.bar(theta, radii, width=width, bottom=0.0, alpha=0.5)

        plt.imsave(d.iloc[0]["id"] + "_directions.png")

    for i in ids:
        random_color = np.random.rand(3)
        plt.plot(df_individuals[i].x_center, df_individuals[i].y_center, color=random_color)
    plt.legend(ids, fontsize='x-small', loc='upper right', ncols=2)
    plt.imsave("paths.png")

    plt.scatter(x = df_summary["total_distance_microns"], y = df_summary["distance_change"])
    plt.imsave("total_distance_vs_change.png")

    fastgif.make_gif(get_frame, 1799, 'tracked_positions.gif', show_progress=True, writer_kwargs={'duration': 0.01})



    