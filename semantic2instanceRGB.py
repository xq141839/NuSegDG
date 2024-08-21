import os
from tqdm import tqdm
import numpy as np
import cv2
import os
from scipy import ndimage as ndi
import random
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from stardist.models import StarDist2D
from stardist.data import test_image_nuclei_2d
from csbdeep.utils import normalize
import tensorflow as tf


model = StarDist2D.from_pretrained('2D_versatile_fluo')
color_path = r"E:\binary_nuclei\density\all4\TNBC_png"
os.makedirs(color_path,exist_ok=True)
save_path = r"E:\binary_nuclei\density\all4\TNBC_npy"
os.makedirs(save_path,exist_ok=True)

image_path = r"E:\Research\results\all4\all4_latest\TNBC"
slide_images = os.listdir(image_path)


skip_images = []

for slide_image in tqdm(slide_images):

    image_name = list(slide_image.split('.'))[0]
    full_image_path = os.path.join(image_path, slide_image)

    mask_ = cv2.imread(full_image_path, cv2.IMREAD_GRAYSCALE)
    
    targets = np.zeros((*mask_.shape[:2], 2))
    instance_idx = mask_ == 255

    if slide_image in skip_images:
        num_labels, labels = cv2.connectedComponents(instance_idx.astype(np.uint8)*255)
        for label in range(1, num_labels):
            targets[..., 0][labels == label] = label
        targets[instance_idx,1] = 1
        targets = targets.astype(int)
    else:
        mask_[mask_> 0] = 1
        labels, _ = model.predict_instances(mask_.astype(float))
        for label in np.unique(labels):
            if label == 0:
                continue
            targets[..., 0][labels == label] = label
        targets[instance_idx,1] = 1
        targets = targets.astype(int)

    drawed_img = np.zeros((mask_.shape[0], mask_.shape[1], 3), dtype=np.uint8)
    inst_map = targets[:,:,0]

    for i in np.unique(labels):
        if i == 0:
            continue
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        drawed_img[inst_map==i] = [b, g, r]

    #drawed_img = ContourDraw(image, targets[:,:,0])
    cv2.imwrite(os.path.join(color_path, slide_image), drawed_img)
    np.save(os.path.join(save_path, slide_image), targets)