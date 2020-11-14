import os

INPUT_DIR = "../datasets/ukbench100"

# Path to output directory
OUTPUT_DIR = "output"

# Path to low/high resolution images
LR_IMAGES = os.path.sep.join([OUTPUT_DIR, "lr"])
HR_IMAGES = os.path.sep.join([OUTPUT_DIR, "hr"])

# Path to low/hight resolution images hdf5 files
LR_HDF5 = os.path.sep.join([OUTPUT_DIR, "lr.hdf5"])
HR_HDF5 = os.path.sep.join([OUTPUT_DIR, "hr.hdf5"])

MODEL_PATH = os.path.sep.join([OUTPUT_DIR, "srcnn.model"])
PLOT_PATH = os.path.sep.join([OUTPUT_DIR, "plot.png"])

# SRCNN parameters
BATCH_SIZE = 128
EPOCHS = 10

SCALE = 2.0

INPUT_DIM = 33
OUTPUT_DIM = 21
PAD = (INPUT_DIM - OUTPUT_DIM) // 2

# Step size of the sliding window
STRIDE = 14
