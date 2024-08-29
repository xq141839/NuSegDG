
import os
from skimage import io, transform, color, img_as_ubyte
import numpy as np
from torch.utils.data import Dataset
import cv2
import torch
import torchvision.transforms as pytorch_transforms
import torch.nn.functional as F
from albumentations.pytorch.transforms import ToTensor
import albumentations as A
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

img_list = os.listdir("E:\Research\medsam_v2\code\datasets\TNBC\mask_1024")

for img_name in tqdm(img_list):
    
    image_path = f'E:\Research\medsam_v2\code\datasets\TNBC\mask_1024\{img_name}'
    img = cv2.imread(image_path, 0)  # Load in grayscale mode
    if img is None:
        print(f"Failed to load image from {image_path}. Check if the file exists and is accessible.")
        continue  # Skip this iteration and move to the next image
    
    img[img < 255] = 0
    target = np.zeros(img.shape, dtype=np.uint8)
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img)
    for i in range(1, num_labels):
        one_nuclei = np.zeros(img.shape, dtype= np.uint8)
        row, col = int(centroids[i][0]), int(centroids[i][1])
        one_nuclei[col][row]= 255
        one_nuclei= 10.0 * (one_nuclei[:,:]>0)
        one_nuclei = gaussian_filter(one_nuclei, sigma=(3,3), mode='constant', radius=10)
        
        normalised =(one_nuclei - one_nuclei.min()) / (one_nuclei.max() - one_nuclei.min())
        normalised *= 128
        normalised[normalised !=0] +=128
        normalised[normalised >255] = 255
        normalised[normalised <129] = 0
        normalised[normalised >0] = 255
        
        target += normalised.astype(np.uint8)
        
    output_path = f"E:\Research\medsam_v2\code\datasets\TNBC\DensityMap\{img_name}"
    cv2.imwrite(output_path, target)