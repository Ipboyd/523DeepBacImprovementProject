import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

import os
import cv2
import numpy as np
import pandas as pd
from data_analysis import count_segments

def augment_images_in_folder(image_folder_path, mask_folder_path, density_factor, output_folder_path = None):
    if not output_folder_path:
        output_folder_path = os.path.dirname(image_folder_path)
        
    # Create main output folder
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
    

    # Create separate folders for augmented images and masks
    augmented_image_folder = os.path.join(output_folder_path, f'augmented_{os.path.basename(image_folder_path)}_dens_{density_factor}')
    augmented_mask_folder = os.path.join(output_folder_path, f'augmented_{os.path.basename(mask_folder_path)}_dens_{density_factor}')
    
    os.makedirs(augmented_image_folder, exist_ok=True)
    os.makedirs(augmented_mask_folder, exist_ok=True)

    augmentation_info = []
    image_files = [f for f in os.listdir(image_folder_path) if f.endswith(('tif','.png', '.jpg', '.jpeg'))]
    
    for image_file in image_files:
        patterns = ['001', '002', '__'] # Since they are already dense we don't want to deal with them
        if not any(pattern in image_file for pattern in patterns):
            mask_file = image_file  # Assuming mask files have the same name as image files
            original_image_path = os.path.join(image_folder_path, image_file)
            mask_image_path = os.path.join(mask_folder_path, mask_file)

            original_image_rgb, mask_image_rgb, final_image, final_mask = process_images(
                original_image_path,
                mask_image_path,
                visualize_individual=False,
                display_plots=True,
                density_factor=density_factor
            )

            if final_image is not None and final_mask is not None:
                augmented_image_path = os.path.join(augmented_image_folder, f'augmented_{image_file}')
                augmented_mask_path = os.path.join(augmented_mask_folder, f'augmented_{mask_file}')

                cv2.imwrite(augmented_image_path, cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR))
                cv2.imwrite(augmented_mask_path, cv2.cvtColor(final_mask, cv2.COLOR_RGB2BGR))

                augmentation_info.append({
                    "original_image": original_image_path,
                    "mask": mask_image_path,
                    "augmented_image": augmented_image_path,
                    "augmented_mask": augmented_mask_path,
                    "density_factor": density_factor
                })

    info_df = pd.DataFrame(augmentation_info)
    info_df.to_csv(os.path.join(output_folder_path, 'augmentation_info.csv'), index=False)
    
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def get_unique_colors_2(mask):
    unique_colors = np.unique(mask.reshape(-1, mask.shape[2]), axis=0)
    unique_colors = unique_colors[~np.all(unique_colors == [0, 0, 0], axis=1)] # Discard black
    return unique_colors

def get_unique_colors(mask):
    if len(mask.shape) == 3:  # Multi-channel mask
        unique_colors = np.unique(mask.reshape(-1, mask.shape[2]), axis=0)
    else:  # Grayscale mask
        unique_colors = np.unique(mask)
    unique_colors = unique_colors[~np.all(unique_colors == [0, 0, 0], axis=-1)]
    return unique_colors


def extract_bacteria(original, mask, unique_colors):
    bacteria_dict = {}
    for color in unique_colors:
        color_mask = np.all(mask == color, axis=-1).astype(np.uint8) * 255
        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            contour = contours[0]
            x, y, w, h = cv2.boundingRect(contour)
            bacteria_image = original[y:y+h, x:x+w].copy()
            bacteria_mask = color_mask[y:y+h, x:x+w]
            bacteria_image[bacteria_mask == 0] = [0, 0, 0]  # Set background to black
            bacteria_dict[tuple(color)] = (bacteria_image, bacteria_mask)
    return bacteria_dict

def create_background(original_image, mask_image, dilation_iterations=5):
    # Create a binary mask where bacteria are white and background is black
    gray = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
    _, binary_mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # Dilate the binary mask to make bacteria areas larger
    kernel = np.ones((5, 5), np.uint16)  # Define a kernel for dilation
    dilated_mask = cv2.dilate(binary_mask, kernel, iterations=dilation_iterations)

    # Calculate the median color of the original image excluding bacteria
    masked_image = original_image.copy()
    masked_image[dilated_mask == 255] = 0  # Set dilated bacteria areas to black

    # Calculate median color of non-bacteria pixels
    median_color = np.median(masked_image[masked_image > 0], axis=0).astype(np.uint16)

    # Create a copy of the original image for modification
    modified_image = original_image.copy()

    # Replace dilated bacteria areas with the median color
    modified_image[dilated_mask == 255] = median_color

    return modified_image

def visualize_individual_bacteria(bacteria_dict):
    n_bacteria = len(bacteria_dict)
    cols = 5
    rows = (n_bacteria + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 4 * rows))
    axes = axes.flatten()
    
    for i, (color, (bacteria_image, _)) in enumerate(bacteria_dict.items()):
        axes[i].imshow(cv2.cvtColor(bacteria_image, cv2.COLOR_BGR2RGB))
        axes[i].set_title(f"Bacteria {i+1}\nColor: {color}")
        axes[i].axis('off')
    
    for i in range(n_bacteria, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

# Define a function to generate distinct colors
def generate_unique_colors(num_colors):
    np.random.seed(0)  # For reproducibility
    return np.random.randint(0, 255, size=(num_colors, 3), dtype=np.uint16)

def rotate_and_crop(image, mask, angle):
    """
    Rotates the bacteria image and mask, and then crops to the smallest bounding box containing the bacteria.
    Args:
        image (numpy.ndarray): Input bacteria image (BGR or grayscale).
        mask (numpy.ndarray): Corresponding mask for the bacteria.
        angle (float): Rotation angle in degrees.

    Returns:
        cropped_rotated_image (numpy.ndarray): Cropped image after rotation.
        cropped_rotated_mask (numpy.ndarray): Cropped mask after rotation.
    """
    # Step 1: Rotate the image and mask
    center = (image.shape[1] // 2, image.shape[0] // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
    rotated_mask = cv2.warpAffine(mask, rotation_matrix, (mask.shape[1], mask.shape[0]))

    # Step 2: Find the bounding box of the rotated mask
    contours, _ = cv2.findContours((rotated_mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None  # No bacteria detected after rotation

    x, y, w, h = cv2.boundingRect(contours[0])  # Bounding box of the first contour

    # Step 3: Crop the rotated image and mask to the bounding box
    cropped_rotated_image = rotated_image[y:y+h, x:x+w]
    cropped_rotated_mask = rotated_mask[y:y+h, x:x+w]

    return cropped_rotated_image, cropped_rotated_mask

def process_images(original_image_path, mask_image_path, visualize_individual=False, display_plots=False, density_factor=1):
    # Load images
    original_image = cv2.imread(original_image_path)
    mask_image = cv2.imread(mask_image_path)
    
    #print(f"Mask dtype: {mask_image.dtype}, min: {mask_image.min()}, max: {mask_image.max()}")


    if original_image is None or mask_image is None:
        print("Error loading images.")
        return

    # Process images
    unique_colors = get_unique_colors(mask_image)
    bacteria_dict = extract_bacteria(original_image, mask_image, unique_colors)
    
    # Generate unique colors for each bacterium
    num_bacteria = len(bacteria_dict)
    distinct_colors = generate_unique_colors(num_bacteria)

    # Increase density of bacteria by duplicating them randomly
    dense_bacteria_dict = {}
    for idx, (color, (bacteria_image, bacteria_mask)) in enumerate(bacteria_dict.items()):
        for i in range(density_factor):
            
            # Randomly rotate and crop the bacteria image
            angle = np.random.uniform(-30, 30)  # Random angle between -30 and 30 degrees
            rotated_bacteria_image, rotated_bacteria_mask = rotate_and_crop(bacteria_image, bacteria_mask, angle)
            
            # Skip if no bacteria detected after rotation
            if rotated_bacteria_image is None or rotated_bacteria_mask is None:
                continue

            # Randomly flip the bacteria image horizontally or vertically
            if np.random.rand() > 0.5:  # Randomly decide to flip
                rotated_bacteria_image = cv2.flip(rotated_bacteria_image, 1)  # Flip horizontally
                rotated_bacteria_mask = cv2.flip(rotated_bacteria_mask, 1)  # Flip mask similarly

            # Randomly position the new bacteria image
            offset_x = np.random.randint(0, original_image.shape[1] - rotated_bacteria_image.shape[1])
            offset_y = np.random.randint(0, original_image.shape[0] - rotated_bacteria_image.shape[0])
            
            # Store the transformed bacteria image and mask with its new position and assigned color
            if idx not in dense_bacteria_dict:
                dense_bacteria_dict[idx] = []
            dense_bacteria_dict[idx].append((rotated_bacteria_image.copy(), rotated_bacteria_mask.copy(), (offset_x, offset_y), distinct_colors[idx]))

    background = create_background(original_image.copy(), mask_image)
    
    final_image, final_mask = place_bacteria_on_background(background.copy(), dense_bacteria_dict)

    # Convert images from BGR to RGB for correct color representation in Matplotlib
    original_image_rgb = cv2.cvtColor(original_image , cv2.COLOR_BGR2RGB)
    mask_image_rgb = cv2.cvtColor(mask_image , cv2.COLOR_BGR2RGB)
    final_rgb = cv2.cvtColor(final_image , cv2.COLOR_BGR2RGB)
    final_mask_rgb = cv2.cvtColor(final_mask , cv2.COLOR_BGR2RGB)
    
    print(f"Count before augmentations: {count_segments(mask_image_rgb, method='color')}")
    print(f"Count after augmentations: {count_segments(final_mask_rgb, method='color')}")

    if display_plots:
        # Display results using Matplotlib in a structured manner
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))

        axes[0,0].imshow(original_image_rgb)
        axes[0,0].set_title(f'Original Image: {os.path.basename(original_image_path)}')
        axes[0,0].axis('off')

        axes[0,1].imshow(mask_image_rgb, cmap ="gray")
        axes[0,1].set_title('Mask Image')
        axes[0,1].axis('off')

        axes[1,0].imshow(final_rgb)
        axes[1,0].set_title(f'Final Image with Increased Bacteria Density by {density_factor}')
        axes[1,0].axis('off')

        axes[1,1].imshow(final_mask_rgb, cmap ="gray")
        axes[1,1].set_title(f'Final Mask Image with Increased Bacteria Density by {density_factor}')
        axes[1,1].axis('off')
        plt.savefig('report_fig_2.png', dpi=300, bbox_inches='tight')
        plt.tight_layout()
        plt.show()
        

    return original_image_rgb, mask_image_rgb, final_rgb, final_mask_rgb

import numpy as np
import cv2
import random

def generate_unique_color(used_colors):
    while True:
        # Generate a random color (ensure non-zero to avoid black regions in mask)
        color = tuple(random.randint(1, 255) for _ in range(3))
        if color not in used_colors:
            used_colors.add(color)
            return color

def place_bacteria_on_background(background, bacteria_dict):
    """
    Places rotated and cropped bacteria images onto the background dynamically using the same dtype as input images.

    Args:
        background (numpy.ndarray): Original background image.
        bacteria_dict (dict): Dictionary of bacteria images, masks, and positions.

    Returns:
        final_image (numpy.ndarray): The updated background with bacteria images placed.
        final_mask (numpy.ndarray): The mask with corresponding bacteria areas colored.
    """
    # Determine dtype of the background image dynamically
    dtype = background.dtype
    
    h_bg, w_bg = background.shape[:2]
    final_image = background.copy().astype(dtype)  # Preserve dtype of input
    final_mask = np.zeros((h_bg, w_bg, 3), dtype=dtype)  # Final mask in same dtype
    used_colors = set()  # Keep track of used colors

    for idx, bacteria_list in bacteria_dict.items():
        for bacteria_image, bacteria_mask, (x_pos, y_pos), color in bacteria_list:
            # Ensure the color is unique
            if color is None or tuple(color) in used_colors:
                color = generate_unique_color(used_colors)
            else:
                used_colors.add(tuple(color))

            h_bac, w_bac = bacteria_image.shape[:2]

            # Ensure that the position does not exceed the background dimensions
            x_pos = min(x_pos, w_bg - w_bac)
            y_pos = min(y_pos, h_bg - h_bac)

            # Define region of interest on the background
            roi = final_image[y_pos:y_pos + h_bac, x_pos:x_pos + w_bac]

            # Create inverse mask for blending (convert to dtype)
            mask_inv = cv2.bitwise_not(bacteria_mask).astype(dtype)

            # Black-out area of bacteria in ROI
            background_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

            # Take only region of bacteria from the bacteria image
            bacteria_fg = cv2.bitwise_and(bacteria_image, bacteria_image, mask=bacteria_mask)

            # Add the foreground and background together (ensure same dtype)
            dst = cv2.add(background_bg, bacteria_fg)

            # Place blended result back into original image
            final_image[y_pos:y_pos + h_bac, x_pos:x_pos + w_bac] = dst

            # Update final mask with colored mask for this bacterium
            colored_mask_part = np.zeros_like(final_mask[y_pos:y_pos + h_bac, x_pos:x_pos + w_bac], dtype=dtype)
            
            # Only update pixels where the current mask is zero (no previous bacterium)
            mask_condition = (bacteria_mask > 0) & (np.all(final_mask[y_pos:y_pos + h_bac, x_pos:x_pos + w_bac] == 0, axis=-1))
            colored_mask_part[mask_condition] = color
            
            final_mask[y_pos:y_pos + h_bac, x_pos:x_pos + w_bac] += colored_mask_part

    return final_image, final_mask
