"""
This program is designed to measure the vacuolar membrane localization rate of Sch9 from microscopy images.
It utilizes the YeaZ program for neural network-based image segmentation. The program processes images to 
generate segmentation masks, which are then used to analyze the localization of Sch9. Quantitative measurements 
are extracted and saved to an output directory, facilitating the study of Sch9 localization patterns.

Expected directory structure:
.
├── input
│   └── ERFxxxx              # Experiment ID
│       └── yymmdd           # Date of experiment
│           ├── strain_01    # Name of the strain
│           │   ├── treatment_01     # Name of the treatment
│           │   │   ├── mask          # Brightfield images used for mask creation
│           │   │   │   ├── strain_01_treatment_01_image_01.tif
│           │   │   │   └── strain_01_treatment_01_image_02.tif
│           │   │   └── measure       # Images for fluorescence measurement
│           │   │       ├── G         # GFP-Sch9 images
│           │   │       │   ├── strain_01_treatment_01_image_01_gfp.tif
│           │   │       │   └── strain_01_treatment_01_image_02_gfp.tif
│           │   │       └── R         # FM4-64 images
│           │   │           ├── strain_01_treatment_01_image_01_fm.tif
│           │   │           └── strain_01_treatment_01_image_02_fm.tif
│           │   └── treatment_02
│           │       ├── mask
│           │       │   ├── strain_01_treatment_02_image_01.tif
│           │       │   └── strain_01_treatment_02_image_02.tif
│           │       └── measure
│           │           ├── G
│           │           │   ├── strain_01_treatment_02_image_01_gfp.tif
│           │           │   └── strain_01_treatment_02_image_02_gfp.tif
│           │           └── R
│           │               ├── strain_01_treatment_02_image_01_fm.tif
│           │               └── strain_01_treatment_02_image_02_fm.tif
│           └── strain_02
│               ├── treatment_01
│               │   ├── mask
│               │   │   ├── strain_02_treatment_01_image_01.tif
│               │   │   └── strain_02_treatment_01_image_02.tif
│               │   └── measure
│               │       ├── G
│               │       │   ├── strain_02_treatment_01_image_01_gfp.tif
│               │       │   └── strain_02_treatment_01_image_02_gfp.tif
│               │       └── R
│               │           ├── strain_02_treatment_01_image_01_fm.tif
│               │           └── strain_02_treatment_01_image_02_fm.tif
│               └── treatment_02
│                   ├── mask
│                   │   ├── strain_02_treatment_02_image_01.tif
│                   │   └── strain_02_treatment_02_image_02.tif
│                   └── measure
│                       ├── G
│                       │   ├── strain_02_treatment_02_image_01_gfp.tif
│                       │   └── strain_02_treatment_02_image_02_gfp.tif
│                       └── R
│                           ├── strain_02_treatment_02_image_01_fm.tif
│                           └── strain_02_treatment_02_image_02_fm.tif
└── unet
    ├── measure_Sch9.py       # This script
    ├── neural_network.py     # Neural network utilities
    ├── segment.py            # Segmentation utilities
    └── weights
        └── weights_budding_BF_multilab_0_1.hdf5  # Pre-trained weights for segmentation
"""

import os
import sys
import cv2
import pandas as pd
import numpy as np
from skimage import exposure
import scipy.ndimage as ndimage
import PIL.Image as Image
import glob
import matplotlib.pyplot as plt

from neural_network import prediction, threshold
from segment import segment

# Define input and output directories
INPUT_DIR = "../input"
OUTPUT_DIR = "../output"

# Function to create masks from images
def make_mask(img_mask_list, output_mask_directory):
    segmentations = []
    for img_mask_path in img_mask_list:
        img_name = return_file_name(img_mask_path)
        img_mask = cv2.imread(img_mask_path, -1)
        img_mask = exposure.equalize_adapthist(img_mask)
        img_mask = img_mask * 1
        pred = prediction(img_mask, False)
        th = threshold(pred)
        segmentation = segment(th, pred, min_distance=8)

        # Remove small segments
        unique, count = np.unique(segmentation, return_counts=True)
        for rem in unique[count <= 30]:
            segmentation[segmentation == rem] = 0

        # Save the resulting mask
        result_mask = Image.fromarray(segmentation)
        result_mask.save(os.path.join(output_mask_directory, img_name+"_mask.tif"))        
        segmentations.append(segmentation)
    return segmentations

# Function to measure values from segmented images
def measure_value(segmentations, img_measure_red_list, img_measure_green_list,
                   control=False):
    # Parameters for image processing
    params = [13, -2, 5, 3, 5]
    BLOCK_SIZE, C, MEDIAN_SIZE, KERNEL_SIZE, N_ITERATION = params

    for (segmentation, img_measure_red_path, img_measure_green_path) in zip(segmentations, img_measure_red_list, img_measure_green_list):
        img_name = return_file_name(img_measure_red_path)
        img_core_name = "_".join(img_name.split("_")[:-1])

        # Initialize lists to store measurement values
        cytosol_value = []
        vacuole_membrane_value = []
        vacuole_lumen_value = []
        mask_size = []
        vm_size = []
        vl_size = []
        v_all_size = []
        
        # DataFrame to store measurements
        df = pd.DataFrame(columns=['maskID', 'cell_size', 'vm_size', 'vl_size', 'v_all_size', 'cytoplasm', 'vacuolar_membrane', 'vacuolar_lumen', 'ratio'])
        outputcsv = os.path.join(OUTPUT_DIR, img_core_name, img_name + ".csv")

        # Read and process the red and green channel images
        img_red = cv2.imread(img_measure_red_path,-1)
        img_red = adapt_contrast(img_red, 0.1)
        img_green = cv2.imread(img_measure_green_path,-1)
        img_masks = mask_convert(segmentation)

        # Initialize plot for contours
        mask_contours = []
        fig = plt.figure(figsize = (img_green.shape[1] / 100, img_green.shape[0] / 100))
        ax = fig.add_subplot()
        output_contours = os.path.join(OUTPUT_DIR, img_core_name, img_name + "_contours.png")

        # Apply adaptive thresholding and image processing
        img_red_thresh = cv2.adaptiveThreshold(img_red, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, BLOCK_SIZE, C)#13,-2
        img_red_thresh_median = cv2.medianBlur(img_red_thresh, MEDIAN_SIZE)
        kernel = np.ones((KERNEL_SIZE, KERNEL_SIZE), np.uint8)
        img_red_thresh_dilate = cv2.dilate(img_red_thresh_median, kernel,iterations=N_ITERATION)
        img_red_thresh_close = cv2.erode(img_red_thresh_dilate,kernel,iterations=N_ITERATION)
        img_red_thresh_fill = ndimage.binary_fill_holes(img_red_thresh_close).astype(np.uint8) * 255
        img_red_vm_lumen = img_red_thresh_fill - img_red_thresh_median

        # Process each mask
        for i in range(0, img_masks.shape[2]):
            mask = img_masks[:,:,i]
            vacuole_membrane = mask * (img_red_thresh_median / 255)
            vacuole_lumen = mask * (img_red_vm_lumen / 255)
            vacuole_all = mask * (img_red_thresh_fill / 255)
            cytosol = mask * ((255 - img_red_thresh_fill) /255)

            # Find and annotate contours
            contours, _ = cv2.findContours(mask.astype("uint8"), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            for k in range(len(contours)):
                mask_contours.append(contours[k])
                ax.text(contours[k][0][0][0], contours[k][0][0][1], i, color="w", size="10")

            # Store the measurement values
            cytosol_value.append(np.nanmean(img_green[np.nonzero(cytosol)]))
            vacuole_membrane_value.append(np.nanmean(img_green[np.nonzero(vacuole_membrane)]))
            vacuole_lumen_value.append(np.nanmean(img_green[np.nonzero(vacuole_lumen)]))

            mask_size.append(np.sum(mask))
            vm_size.append(np.sum(vacuole_membrane))
            vl_size.append(np.sum(vacuole_lumen))
            v_all_size.append(np.sum(vacuole_all))
        
        # Calculate ratio and save measurements
        ratio_vacuole2cytosol = np.array(vacuole_membrane_value) / np.array(cytosol_value)
        df["maskID"] = range(img_masks.shape[2])
        df["cell_size"] = mask_size
        df["vm_size"] = vm_size
        df["vl_size"] = vl_size
        df["v_all_size"] = v_all_size
        df["cytoplasm"] = cytosol_value
        df["vacuolar_membrane"] = vacuole_membrane_value
        df["vacuolar_lumen"] = vacuole_lumen_value
        df["ratio"] = ratio_vacuole2cytosol
        
        # Save the measurements to a CSV file
        df.to_csv(outputcsv, index=False)

        # Draw and save contours on the green channel image
        blended = cv2.drawContours(adapt_contrast(img_green, 0.1), mask_contours, -1, (0, 255, 0), 1)
        plt.axis("off")
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        plt.imshow(blended)
        fig.savefig(output_contours)
        plt.close()

# Function to extract the file name from a given path
def return_file_name(path):
    return os.path.splitext(os.path.basename(path))[0]

# Function to convert segmentation masks into individual cell masks
def mask_convert(segmentation_mask):

    image_height = segmentation_mask.shape[0]
    image_width = segmentation_mask.shape[1]
    unique = np.unique(segmentation_mask)
    n_cell = len(unique)
    converted_mask = np.zeros((image_height, image_width, n_cell-1))

    for i in range(n_cell-1):
        converted_mask[:, :, i] = np.where(segmentation_mask == unique[i+1], 1, 0)
        kernel = np.ones((3,3), np.uint8)
        erosion = cv2.erode(converted_mask[:,:,i], kernel, iterations=5)
        converted_mask[:, :, i] = erosion
    
    return converted_mask

# Function to adjust the contrast of an image
def adapt_contrast(img, saturation):
    img = img.astype(np.uint32)
    min_value = np.min(img).astype(np.uint32)
    max_value = np.percentile(img, 100-saturation)
    adjusted_img = 255 * (img.astype(np.uint32)-min_value) / (max_value-min_value)
    std_img = np.where(adjusted_img>255, 255, adjusted_img)
    return np.uint8(std_img)         

# Main function to process all image files
def main(input_directory, output_directory):
    # Extract the path list from command line arguments
    path_arg = sys.argv[1:]
    path_list = ["/".join(path.split("/")[-2:]) for path in path_arg]

    for path in path_list:
        strain_path = glob.glob(os.path.join(input_directory, path, "*"))
        strains = [os.path.basename(path) for path in strain_path]
        treatment_path = [glob.glob(os.path.join(path, "*")) for path in strain_path]
        treatments = list(set([os.path.basename(path) for path in sum(treatment_path, [])]))
        
        for strain in strains:
            for treatment in treatments:
                treatment_path = os.path.join(path, strain, treatment)
                img_mask_list = sorted(glob.glob(os.path.join(input_directory, treatment_path, "mask", "*.tif")))
                img_measure_red_list = sorted(glob.glob(os.path.join(input_directory, treatment_path, "measure", "R", "*.tif")))
                img_measure_green_list = sorted(glob.glob(os.path.join(input_directory, treatment_path, "measure", "G", "*.tif")))

                # Ensure the output directories exist
                os.makedirs(os.path.join(output_directory, treatment_path, "mask"), exist_ok=True)
                for img in img_measure_green_list:
                    img_name = os.path.splitext(os.path.basename(img))[0]
                    img_core_name = "_".join(img_name.split("_")[:-1])
                    os.makedirs(os.path.join(output_directory, treatment_path, "measure", img_core_name), exist_ok=True)

                # Generate segmentation masks and measure values
                segmentations = make_mask(img_mask_list, os.path.join(output_directory, treatment_path, "mask"))
                measure_value(segmentations, img_measure_red_list, img_measure_green_list)

# Entry point of the script
if __name__ == "__main__":
    main(INPUT_DIR, OUTPUT_DIR)