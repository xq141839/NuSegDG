import os
import numpy as np
import cv2
from tqdm import tqdm

# Define the path to the ground truth binary masks
ground_truth_path = r"E:\Research\medsam_v2\code\datasets\CyoNuSeg\mask_1024"
# Define the path to save the .npy files
save_path = r"E:\binary_nuclei\density\allnocyonuseg\gt_npy"
os.makedirs(save_path, exist_ok=True)

# Get all the mask files in the ground truth directory
mask_files = os.listdir(ground_truth_path)

for mask_file in tqdm(mask_files):
    # Construct full path to the mask file
    full_mask_path = os.path.join(ground_truth_path, mask_file)
    # Read the binary mask image
    mask_ = cv2.imread(full_mask_path, cv2.IMREAD_GRAYSCALE)
    
    # Ensure the mask is binary (0 or 255)
    mask_[mask_ > 0] = 1

    # Perform connected component analysis to label instances
    num_labels, labels = cv2.connectedComponents(mask_)
    
    # Initialize targets array with shape (height, width, 2)
    targets = np.zeros((*mask_.shape, 2), dtype=np.int32)
    # Set the first channel to instance labels
    targets[..., 0] = labels
    # Set the second channel to binary mask
    targets[..., 1] = (mask_ > 0).astype(int)
    
    # Save the targets array as a .npy file with the new naming convention
    np.save(os.path.join(save_path, mask_file + '.npy'), targets)

print("Conversion complete.")