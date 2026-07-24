# NematostellaTracking
Code used to process data and extract insights on video data of various life cycle stages of the starlet sea anemone *Nematostella vectensis*.

## Main Components: Things to Know
The key portions of this repository are:
1. A web interface that allows for computer-vision assisted labeling of images segmented from a video, 
2. A pipeline that takes labeled data (or generates its own predictions based on a YOLO model) that automatically tracks organisms and extracts insights, and
3. A notebook that helps you to make parameter files. 

## Web Interface
We want to use computer vision object detection to better track individuals throughout a video (or rather, a stack of images for the video's frames). However, object detection is not perfect, and even though modern models are powerful, small mistakes can magnify over hundreds of images. Therefore, we created a web interface that allows users to manually label and (critically) correct model mistakes. 

Included in the `web_interface` folder, our web interface was built and tested on Python 3.13.1 and allows for users to manual label image data with an existing YOLO model on their local machines. To use the web interface, you will need to make a conda environment with the following packages:
- `nicegui` for the main web interface dependencies
- `tkinter` for file selection. Note that you can specify directories and files with a prewritten path, but this package makes selecting a directory in your local machine's file structure easier
- `pandas` for dataframe management
- optionally, `ultralytics` and `pytorch` for if you wish to utilize a pretrained YOLO model for semi-automated analysis.
- optionally, `PIL` (the modern fork of Pillow), which is used to make the favicon. If you have `ultralytics` installed, it is highly likely you also have this. 

Once your conda environment is activated, the web interface can be launched with the command `python3 main.py`, provided you have navigated to the directory contaning `main.py`. You will be prompted with two text fields; one for a directory of your images, and the other for a path to a pretrained YOLO model, which is optional. Once images are loaded, you can label bounding boxes of planula larvae and polyps as you see fit. New annotations can be created by dragging the mouse from the top-left corner to the bottom-right corner of the annotation, and existing annotations can be modified by clicking on them, after which a secondary menu of reassignment or deletion will appear for that box. Note that when working with a directory, intermediary results are often saved in the file `annotations.csv`, which is automatically loaded when you restart the app and work in the same directory. The "Export CSV" button can be used to ensure exporting of a specific state without relying on autosaves.

While the web interface can be used without knowing any of its other features, the web interface also (by default) remembers the annotations that you delete and automatically deletes YOLO predictions that are similar to those deleted boxes. Similarly, you can also have the interface remember a manual annotation that can get copied to other images if no similar annotation is predicted. Furthermore, a confidence slider to filter out certain automatically-generated annotations is provided. All these aim to make manual labeling easier by reducing the burden of correcting highly similar mistakes from the YOLO model. 

A tutorial video is in progress and will be provided here in the future.

**NOTE**: There is currently an issue with the use of `tkinter` to select a directory and model that drastically hinders interface performance. For the moment, please opt to type the full directory instead while this issue is resolved.

## The Pipeline and Parameter Generation

Included in the `pipeline` folder, our analysis pipeline consists of a notebook for preprocessing videos, which should be done with human supervision (hence the notebook), and all the scripts necessary for automatic analysis of image data. 

Our pipeline was built and tested on Python 3.13.1. The required packages and dependencies are listed as follows: 
- `opencv`, listed as `cv2` in the notebook, for segmenting videos and doing the bulk of the processing
- `numpy`, for manipulating image indexes
- `matplotlib.pyplot`, for visualization in-notebook
- `yaml`, for processing YAML files
- `ultralytics`, for high-level use of YOLO26; this requires `pytorch` as well
- `pandas` for working with dataframes and arrays

The pipeline can be run by navigating to the pipeline folder and running the following command:

`bash main.sh <parameter_file>`

where `<parameter_file>` is the path to your YAML file of parameters. These parameters allow for some customization of tolerance levels and filenames, but the most important parameters are the image directory and a reference height and width in pixels that can be compared to a distance in microns. For our data, we used an 18mm diameter petri dish, hence we only specify one dimension of micron distance. See `parameters/yaml_maker.ipynb` for more details on all the parameters.

# Experiments and Past Code
Below here is prior information that is not necessary to know for running the primary components of this repository; however, they are here for reference regarding all other materials in the repository, used during experimentation and building of the pipeline itself.

There are several stages for this data analysis pipeline. To see the dependencies for each of the stages, see the dedicated sections below.
1. **Video Segmentation and Image Preprocessing**: this involves separating the individual frames of the video, cropping them to only include the portion of interest, and converting to grayscale for ease of memory. (COMPLETED)
2. **Object Detection**: done with YOLO26, this involves taking each of the processed images and detecting polyp and planula life cycle stages. The result of this should be a CSV containing relevant position and class information for an entire image stack. (IN PROGRESS)
3. **Individual Tracking**: done with linear sum assignment, this takes the CSV from the above and reconstructs the path an individual takes, as well as inferring speed, size, and direction using micrometer units. (COMPLETED)

Object detection, critically, is not perfect. It can sometimes miss objects or detect extra objects where there is nothing. To circumvent this, this repository features a web-based interface that allows for you to use an existing YOLO model for semi-automated labeling while also being able to correct any mistakes the model makes. 

## Video Segmentation, Image Preprocessing
This is done with the notebook `video_analysis/initial_processing.ipynb` using Python 3.9.23. The required packages and dependencies are listed as follows:
- `opencv`, listed as `cv2` in the notebook, for segmenting videos and doing the bulk of the processing
- `numpy`, for manipulating image indexes
- `matplotlib.pyplot`, for visualization in-notebook
- `os`, for listing files in directories

Note that to crop an image, the boundaries will need to be determined manually. The boundaries for all videos we record are given in `video_analysis/crop_boundaries.txt`.

## Object Detection
The training of the model and initial creation of functions to extract information is done with the notebook `video_analysis/yolo_initial_modeling.ipynb`. The actual notebook with streamlined processing of an image stack is `video_analysis/model_result_extraction.ipynb`. Both are run on Python 3.13.1. The required packages and dependencies are listed as follows:
- `ultralytics`, for high-level use of YOLO26; this requires `pytorch` as well
- `matplotlib`, `PIL`, `requests`, `opencv`, and `shutil` for working with images
- `os` for listing files in directories
- `pandas` and `numpy` for working with dataframes and arrays
- `yaml` for working with YAML files

## Tracking
The notebook `position_tracking/tracking.ipynb` can be used to take the detection results from the previous phase and assign IDs to each detected individual. The required packages are listed as follows:
- `pandas` and `numpy` for working with dataframes and arrays
- `scipy.optimize` for linear sum assignment; this is done with a modified Jonker-Volgenant algorithm (see `scipy` documentation, https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html).

You can also reconstruct paths and make animated GIFs of the object detection results. For these, you will need the following packages:
- `matplotlib` and `opencv` for making plots
- `fastgif` for making animated GIFs; this relies on the dependency `framemaker.py` for use in a Jupyter notebook