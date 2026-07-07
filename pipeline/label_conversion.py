import pandas as pd
import numpy as np
import os
import sys
import yaml
import warnings

from PIL import Image
from tqdm import tqdm

def get_image_size(filename, size_cache):
    if filename in size_cache:
        return size_cache[filename]
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Image not found: {filename}")
    with Image.open(filename) as img:
        w, h = img.size
    size_cache[filename] = (w, h)
    return w, h

if __name__ == "__main__":
    print("Converting CSV labels to folder of .txt files.")

    with open(sys.argv[1], "r") as fr:
        params = yaml.load(fr, yaml.Loader)

    if params["show_warnings"] == False:
        warnings.filterwarnings('ignore')
    
    os.chdir(params["image_dir"])
    os.makedirs("labels", exist_ok=True)

    df = pd.read_csv(params["labels"])

    #section is run if annotations.csv is from the web interface, where the dataframe has a yolo_class
    if "yolo_class" in df.columns:
        df = df.filter(items = ["filename", "class", "x1", "y1", "x2", "y2"])
        df.columns = ["image_filename", "class", "x1", "y1", "x2", "y2"] #rename for consistency
        classes = []
        for i in range(len(df)):
            if "Polyp" in df.iloc[i]["class"]:
                classes.append(1)
            elif "Planula" in df.iloc[i]["class"]:
                classes.append(0)
            else:
                print("Warning: Image " + df.iloc[i]["image_filename"] + " has an improperly named label")

        df["class"] = classes

    files = df["image_filename"].unique()
    df_files = []
    for f in files:
        df_files.append(df[df.image_filename == f])

    size_cache = {}
    n_images = 0
    n_boxes = 0
    n_skipped_images = 0

    for i in tqdm(range(len(df_files))):
        filename = df_files[i].iloc[0]["image_filename"]

        try:
            img_w, img_h = get_image_size(filename, size_cache)
        except FileNotFoundError as e:
            print("WARNING: " + filename + " has no corresponding image found")
            n_skipped_images += 1
            continue

        lines = []
        for j in range(len(df_files[i])):
            cls = df_files[i].iloc[j]["class"]
            x1 = df_files[i].iloc[j]["x1"]
            y1 = df_files[i].iloc[j]["y1"]
            x2 = df_files[i].iloc[j]["x2"]
            y2 = df_files[i].iloc[j]["y2"]

            # Ensure correct ordering (in case x1>x2 or y1>y2)
            x_min, x_max = sorted((x1, x2))
            y_min, y_max = sorted((y1, y2))

            if x_min < 0 or y_min < 0 or x_max > img_w or y_max > img_h:
                raise ValueError(
                    f"Box out of image bounds for {filename}: "
                    f"({x1},{y1},{x2},{y2}) vs image size ({img_w},{img_h}). "
                )

            box_w = x_max - x_min
            box_h = y_max - y_min
            x_center = x_min + box_w / 2
            y_center = y_min + box_h / 2

            # Normalize
            x_center_n = x_center / img_w
            y_center_n = y_center / img_h
            box_w_n = box_w / img_w
            box_h_n = box_h / img_h

            lines.append(f"{cls} {x_center_n:.6f} {y_center_n:.6f} {box_w_n:.6f} {box_h_n:.6f}")
            n_boxes += 1

        out_file = "labels/" + filename + ".txt"
        with open(out_file, "w") as fw:
            fw.write("\n".join(lines))
            if lines:
                fw.write("\n")

        n_images += 1

    print("Labels saved to folder \"labels\"")
    if n_skipped_images:
        print("Skipped " + n_skipped_images + " image(s) due to missing files.")