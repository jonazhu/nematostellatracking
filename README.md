# NematostellaTracking
Code used to process data and extract insights on video data of various life cycle stages of the starlet sea anemone *Nematostella vectensis*.

There are several stages for this data analysis pipeline. To see the dependencies for each of the stages, see the dedicated sections below.
1. **Video Segmentation and Image Preprocessing**: this involves separating the individual frames of the video, cropping them to only include the portion of interest, and converting to grayscale for ease of memory. (COMPLETED)
2. **Object Detection**: done with YOLO26, this involves taking each of the processed images and detecting polyp and planula life cycle stages. The result of this should be a CSV containing relevant position and class information for an entire image stack. (IN PROGRESS)
3. **Individual Tracking**: done with linear sum assignment, this takes the CSV from the above and reconstructs the path an individual takes, as well as inferring speed, size, and direction using micrometer units. (IN PROGRESS)

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
This is currently still being improved. The notebook `position_tracking/tracking.ipynb` can be used to take the detection results from the previous phase and assign IDs to each detected individual. The required packages are listed as follows:
- `pandas` and `numpy` for working with dataframes and arrays
- `scipy.optimize` for linear sum assignment; this is done with a modified Jonker-Volgenant algorithm (see `scipy` documentation, https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html).

You can also reconstruct paths and make animated GIFs of the object detection results. For these, you will need the following packages:
- `matplotlib` and `opencv` for making plots
- `fastgif` for making animated GIFs; this relies on the dependency `framemaker.py` for use in a Jupyter notebook
