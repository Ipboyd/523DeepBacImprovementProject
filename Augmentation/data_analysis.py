import os
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import math

import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import math

def display_tif_images(brightfield_folder, masks_folder, display_percentage=10):
    # Get all .tif files
    brightfield_files = [f for f in os.listdir(brightfield_folder) if f.endswith('.tif')]
    mask_files = [f for f in os.listdir(masks_folder) if f.endswith('.tif')]

    # Match brightfield images with masks (assume they have the same filenames)
    common_files = sorted(list(set(brightfield_files) & set(mask_files)))
    total_images = len(common_files)

    print(f"Total number of images: {total_images}")
    
    # Calculate number of images to display based on percentage
    num_to_display = math.ceil(total_images * display_percentage / 100)
    print(f"Displaying {num_to_display} images ({display_percentage}% of total).")

    # Plot brightfield and mask images
    plt.figure(figsize=(10, 2 * num_to_display))
    for i, filename in enumerate(common_files[:num_to_display]):
        brightfield_path = os.path.join(brightfield_folder, filename)
        mask_path = os.path.join(masks_folder, filename)

        # Load images
        brightfield_image = cv2.imread(brightfield_path, cv2.IMREAD_UNCHANGED)
        mask_image = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

        # Show brightfield image
        plt.subplot(num_to_display, 2, i * 2 + 1)
        plt.imshow(brightfield_image, cmap='gray')
        plt.title(f"Brightfield: {filename}", fontsize=8)
        plt.axis('off')

        # Show corresponding mask
        plt.subplot(num_to_display, 2, i * 2 + 2)
        plt.imshow(mask_image, cmap='gray')
        plt.title(f"Mask: {filename}, count: {count_segments(np.array(mask_image.copy()), method='color')}", fontsize=8)
        plt.axis('off')

    # Tight layout to remove gaps
    plt.tight_layout()
    plt.show()
    
    
def get_metadata(brightfield_folder, masks_folder):
     
    # Get all .tif files
    brightfield_files = [f for f in os.listdir(brightfield_folder) if f.endswith('.tif')]
    mask_files = [f for f in os.listdir(masks_folder) if f.endswith('.tif')]

    # Match brightfield images with masks (assume they have the same filenames)
    common_files = sorted(list(set(brightfield_files) & set(mask_files)))
    total_images = len(common_files)
    
    # Create a list to store metadata for the table
    metadata = []
    
    for i, filename in enumerate(common_files):
        brightfield_path = os.path.join(brightfield_folder, filename)
        mask_path = os.path.join(masks_folder, filename)
        
        # Load images
        brightfield_image = cv2.imread(brightfield_path, cv2.IMREAD_UNCHANGED)
        mask_image = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        
        # Add metadata for the table
        metadata.append({
            "Image Name": filename,
            "Brightfield Shape": brightfield_image.shape,  # (width, height)
            "Mask Shape": mask_image.shape,                # (width, height)
            "Count": count_segments(np.array(mask_image), method='color'),
            "Dtype": {"Image": brightfield_image.dtype, "mask": mask_image.dtype},
        })
        

    # Create a table using pandas
    metadata_df = pd.DataFrame(metadata)
    return metadata_df


# Colored ROI

import numpy as np
from skimage import io, measure
import cv2

def count_segments(mask, method='color'):
    """
    Count segments in a mask image using the specified method.
    
    Parameters:
    mask_path (str): Path to the mask image file.
    method (str): Method to use for counting. Options are 'color', 'label', or 'contour'.
    
    Returns:
    int: Number of segments counted.
    """
    
    if method == 'color':
        # Count unique colors
        if mask.ndim == 2:  # If grayscale
            unique_colors = np.unique(mask)
        else:  # If color
            unique_colors = np.unique(mask.reshape(-1, mask.shape[2]), axis=0)
        num_segments = len(unique_colors) - 1  # Subtract 1 to exclude background
        
    elif method == 'label':
        # Use connected component labeling
        if mask.ndim > 2:  # If color, convert to grayscale
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        labeled_mask = measure.label(mask)
        num_segments = labeled_mask.max()
        
    elif method == 'contour':
        # Use contour detection
        if mask.ndim > 2:  # If color, convert to grayscale
            gray = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        else:
            gray = mask
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        num_segments = len(contours)
        
    else:
        raise ValueError("Invalid method. Choose 'color', 'label', or 'contour'.")
    
    return num_segments

import plotly.express as px

def plot_count_histogram(metadata_df, title="Distribution of Counts"):
    """
    Creates an interactive histogram of the 'Count' column in the metadata DataFrame.
    
    Args:
        metadata_df (pd.DataFrame): DataFrame containing image metadata with a 'Count' column.
        title (str): Title of the histogram.
    """
    if 'Count' not in metadata_df.columns:
        print("The DataFrame does not have a 'Count' column.")
        return

    # Create histogram using Plotly
    fig = px.histogram(
        metadata_df, 
        x='Count', 
        title=title, 
        labels={'Count': 'Number of Segments (Bacteria Count)'},
        nbins=20,  # Adjust the number of bins as needed
        template='plotly_white'
    )
    
    # Customize layout
    fig.update_layout(
        xaxis_title="Bacteria Count",
        yaxis_title="Frequency (# of images)",
        bargap=0.1
    )
    
    # Show the plot
    fig.show()
    
import os

def create_folder_path(root_folder, *subfolders):
    """
    Creates and returns a folder path dynamically.

    Args:
        root_folder (str): The base directory.
        *subfolders (str): Additional subfolders e.g., ('train', 'test'.), ('full_images', 'brightfield').

    Returns:
        str: The dynamically created folder path.
    """
    # Combine root, and additional subfolders
    full_path = os.path.join(root_folder, *subfolders)
    
    # Create the folder if it doesn't exist
    os.makedirs(full_path, exist_ok=True)
    
    return full_path


def combine_and_plot_histogram(folder_mask_pairs, title="Distribution of Counts After Augmentation"):
    """
    Combines images and masks from multiple folder-mask pairs, calculates counts,
    and creates a combined histogram using Plotly.

    Args:
        folder_mask_pairs (list of tuples): List of (image_folder, mask_folder) pairs.
        title (str): Title of the histogram.

    Returns:
        pd.DataFrame: Combined metadata DataFrame.
    """
    combined_metadata = []

    for image_folder, mask_folder in folder_mask_pairs:
        image_files = [f for f in os.listdir(image_folder) if f.endswith(('tif', 'png', 'jpg', 'jpeg'))]
        mask_files = [f for f in os.listdir(mask_folder) if f.endswith(('tif', 'png', 'jpg', 'jpeg'))]

        # Match images and masks by filenames
        common_files = sorted(list(set(image_files) & set(mask_files)))

        for file in common_files:
            image_path = os.path.join(image_folder, file)
            mask_path = os.path.join(mask_folder, file)

            # Read mask
            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            if mask is None:
                print(f"Skipping {file} as the mask could not be read.")
                continue

            # Calculate count using count_segments
            count = count_segments(mask, method="color")  # Ensure count_segments is implemented

            # Append metadata
            combined_metadata.append({
                "Image": image_path,
                "Mask": mask_path,
                "Count": count
            })

    # Convert combined metadata to DataFrame
    metadata_df = pd.DataFrame(combined_metadata)

    # Plot the histogram
    plot_count_histogram(metadata_df, title)

    return metadata_df
