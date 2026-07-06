import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import yaml
import os
import io
from PIL import Image, ImageDraw

import warnings
warnings.filterwarnings('ignore')

#TODO:
#clean general structure
#integrate hyperparameter processing
#add one more type of animated GIF that overlays paths as they are being made - will experiment in notebook first
#integrate legend into scatterplot
#figure out how to make good plots for speed, distance - all in one or separate histograms for each?

def get_frame_old(idx, df_files):
    img_current = plt.imread(df_files[idx].iloc[0]["image_filename"])
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

    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    return Image.open(buf)

def get_frame(idx, df_files):
    img_current = Image.open(df_files[idx].iloc[0]["image_filename"])
    img_current = img_current.convert('RGB')
    draw = ImageDraw.Draw(img_current)
    for i in range(len(df_files[idx])):
        x1 = df_files[idx].iloc[i]["x1"]
        y1 = df_files[idx].iloc[i]["y1"]
        x2 = df_files[idx].iloc[i]["x2"]
        y2 = df_files[idx].iloc[i]["y2"]
        if df_files[idx].iloc[i]["class"] == 0.0 or df_files[idx].iloc[i]["class"] == "Planula":
            draw.rectangle((x1, y1, x2, y2), outline="blue")
        else: 
            draw.rectangle((x1, y1, x2, y2), outline="red")

    return img_current

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
    print("Beginning diagram creation.")

    with open(sys.argv[1], "r") as fr:
        params = yaml.load(fr, yaml.Loader)

    if params["show_warnings"] == False:
        warnings.filterwarnings('ignore')
            
    os.chdir(params["image_dir"])

    df = pd.read_csv(params["tracked_labels"])

    files = df["image_filename"].unique()
    df_files = []
    for f in files:
        df_files.append(df[df.image_filename == f])

    ids = df["id"].unique()
    df_individuals = []
    for i in ids:
        df_individuals.append(df[df.id == i])

    print("Creating animated GIF of tracked positions.")

    frames = []
    for i in range(len(df_files)):
        frames.append(get_frame(i, df_files))

    frames[0].save('positions_tracked.gif',
               save_all=True, append_images=frames[1:], optimize=False, duration=40, loop=0)
    print("Tracked positions GIF saved to positions_tracked.gif")

    # for d in df_individuals:
    #     N = 8
    #     theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
    #     radii = get_direction_numbers(d)
    #     width = np.pi / 4 * np.ones(8)

    #     colors = plt.colormaps["viridis"](radii / 10.)

    #     ax = plt.subplot(projection='polar')
    #     ax.bar(theta, radii, width=width, bottom=0.0, alpha=0.5)

    #     plt.imsave(d.iloc[0]["id"] + "_directions.png")

    # for i in ids:
    #     random_color = np.random.rand(3)
    #     plt.plot(df_individuals[i].x_center, df_individuals[i].y_center, color=random_color)
    # plt.legend(ids, fontsize='x-small', loc='upper right', ncols=2)
    # plt.imsave("paths.png")

    # plt.scatter(x = df_summary["total_distance_microns"], y = df_summary["distance_change"])
    # plt.imsave("total_distance_vs_change.png")

    # fastgif.make_gif(get_frame, 1799, 'tracked_positions.gif', show_progress=True, writer_kwargs={'duration': 0.01})

