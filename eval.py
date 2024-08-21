import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))


import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
from dataloader import BinaryLoader
from skimage import measure, morphology
import albumentations as A
from albumentations.pytorch import ToTensor
import argparse
import time
import pandas as pd
import cv2
import os
from skimage import io, transform
from PIL import Image
import json
from tqdm import tqdm
import torch.nn.functional as F
from typing import Any, Dict, List, Tuple
from model import SAMB
from automatic_mask_generator import SamAutomaticMaskGenerator
from functools import partial
from scipy import ndimage as ndi
from monai.metrics import compute_hausdorff_distance, compute_percent_hausdorff_distance, HausdorffDistanceMetric
from monai.metrics import DiceMetric, compute_iou
from monai.metrics import HausdorffDistanceMetric
from monai.transforms import EnsureType
from monai.data import decollate_batch
import matplotlib.pyplot as plt
from monai.transforms import AsDiscrete

def compute_precision(pred, target):
    pred = pred.view(-1)  # Flatten the tensor
    target = target.view(-1)  # Flatten the tensor
    tp = ((pred == 1) & (target == 1)).sum().float()
    fp = ((pred == 1) & (target == 0)).sum().float()
    precision = tp / (tp + fp + 1e-8)
    return precision

def compute_recall(pred, target):
    pred = pred.view(-1)  # Flatten the tensor
    target = target.view(-1)  # Flatten the tensor
    tp = ((pred == 1) & (target == 1)).sum().float()
    fn = ((pred == 0) & (target == 1)).sum().float()
    recall = tp / (tp + fn + 1e-8)
    return recall

def merge_masks(masks, save_dir, img_id):

   
    assert len(masks.shape) == 5
    batch_size, num_masks, channels, height, width = masks.shape
    assert channels == 1
    
    
    merged_mask = torch.zeros((batch_size, height, width), device=masks.device, dtype=torch.uint8)
    
    for b in range(batch_size):
        for n in range(num_masks):
            
            mask_np = masks[b, n, 0].cpu().numpy()
            mask_np = (mask_np > 0.5).astype(np.uint8)

            # cv2.imwrite(os.path.join(save_dir, f'{img_id}_mask_{n}.png'), mask_np * 255)

            
            merged_mask[b] = merged_mask[b] | torch.tensor(mask_np, device=masks.device, dtype=torch.uint8)

    merged_mask = merged_mask.float()
    
    return merged_mask

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='CyoNuSeg',type=str, help='TNBC')
    parser.add_argument('--jsonfile', default='data_split.json',type=str, help='')
    parser.add_argument('--size', type=int, default=1024, help='epoches')
    parser.add_argument('--model',default='../code/outputs/8.13Renew_all_epoch_14.pth', type=str, help='')
    parser.add_argument('--prompt',default='point', type=str, help='box, point, auto')
    args = parser.parse_args()
    
    save_png = f'visual/{args.dataset}/'

    os.makedirs(save_png,exist_ok=True)

    args.jsonfile = f'datasets/{args.dataset}/data_split.json'
    
    
    print(f"Loading JSON file from: {args.jsonfile}")
    
    
    with open(args.jsonfile, 'r') as f:
        df = json.load(f)

    test_files = df['test']
    test_dataset = BinaryLoader(args.dataset, test_files, A.Compose([
                                        A.Resize(args.size, args.size),
                                        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                                        ToTensor()
                                        ]))

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1)
    

    model = SAMB(img_size=args.size)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")

        model = nn.DataParallel(model)
    
    model.load_state_dict(torch.load(args.model), strict=True)
    model.to(device='cuda')
    
    ###################################
 
    computeHD = HausdorffDistanceMetric(include_background=False, reduction="mean", get_not_nans=False)
    computeDice = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)


    avgIoU = []
    avgHD = []
    avgDice = []
    avgRec = [] # Recall
    avgPrec = [] # Precision

    ###################################


    
    with torch.no_grad():
        
        
        for _, img, mask, img_id, density_map in tqdm(test_loader):
            img = Variable(img).cuda()
            mask = Variable(mask).cuda()
            density_map = Variable(density_map).cuda()

            print(img.shape)
            
            if args.prompt in ["point", "box"]:
                if args.prompt == "point":
                    mask_pred = model(img, mask=density_map, mode='point')[0]
                elif args.prompt == "box":
                    mask_pred = model(img, mask=mask, mode='box')
                    #print("mask_pred shape:", mask_pred.shape)
                    mask_pred = merge_masks(mask_pred, save_png, img_id[0])

                unique, counts = torch.unique(mask_pred, return_counts=True)
                print("Unique values in the mask prediction:", unique)
                print("Counts of each value:", counts)
                
                                
                mask_pred = torch.round(mask_pred)
                mask_pred = (mask_pred > 0).float()
                mask_pred = mask_pred.unsqueeze(1)
                mask = torch.round(mask)
                mask = (mask > 0).float()
                
                unique, counts = torch.unique(mask_pred, return_counts=True)
                print("Unique values in the mask prediction11:", unique)
                print("Counts of each value11:", counts)

                
                mask_pred_squeezed = mask_pred.squeeze()
                mask_squeezed = mask.squeeze()
                
                print("mask_pred shape:", mask_pred.shape)
                print("mask shape:", mask.shape)
                
                mask_draw = mask_pred.clone().detach()
                gt_draw = mask.clone().detach()
            
                plt.figure(figsize=(12, 6))
                plt.subplot(1, 2, 1)
                plt.imshow(mask_draw.cpu().squeeze(), cmap='gray')
                plt.title('Predicted Mask')
                plt.axis('off')

                plt.subplot(1, 2, 2)
                plt.imshow(gt_draw.cpu().squeeze(), cmap='gray')
                plt.title('Ground Truth Mask')
                plt.axis('off')

                plt.show()
                
            
                # calculate metrics
                iou = compute_iou(mask_pred, mask, include_background=False, ignore_empty=True).mean().item()
                dice = computeDice(mask_pred, mask).item()
                hd = computeHD(mask_pred, mask).item()
                recall = compute_recall(mask_pred_squeezed, mask_squeezed).item()
                precision = compute_precision(mask_pred_squeezed, mask_squeezed).item()

                avgDice.append(dice)
                avgIoU.append(iou)
                avgHD.append(hd)
                avgRec.append(recall)
                avgPrec.append(precision)

                ###################################

                mask_numpy = mask_draw.cpu().detach().numpy()[0][0]
                mask_numpy[mask_numpy == 1] = 255 
                img_id = img_id[0].split('.')[0]
                cv2.imwrite(f'{save_png}{img_id}.png', mask_numpy)
            
                
                torch.cuda.empty_cache()
    

            elif args.prompt == "auto":
                mask_generator = SamAutomaticMaskGenerator(model)


                img_cv2 = cv2.imread(f'datasets/{args.dataset}/image_1024/{img_id[0]}.png')
                img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)         
            

                mask_list = mask_generator.generate(img_cv2)
                mask_pred = np.zeros((args.size, args.size))

                if len(mask_list) > 0:
                
                    sorted_anns = sorted(mask_list, key=(lambda x: x['area']), reverse=True)

                    for ann in sorted_anns:
                        m = ann['segmentation']
                        mask_pred = np.maximum(mask_pred, m)

                    ###################################

                mask_pred = torch.tensor(mask_pred).cuda()
                mask_pred = mask_pred.unsqueeze(0).unsqueeze(0)
                mask = mask.unsqueeze(0).unsqueeze(0)
                
                if mask.shape[-2:] != (args.size, args.size):
                    mask = F.interpolate(mask, size=(args.size, args.size), mode='nearest')
                mask = mask.squeeze(0)
                
                #binarize
                mask_pred = torch.round(mask_pred)
                mask_pred = (mask_pred > 0).float()
                mask = torch.round(mask)
                mask = (mask > 0).float()

                
                
                mask_pred_squeezed = mask_pred.squeeze()
                mask_squeezed = mask.squeeze()
                
                
                mask_draw = mask_pred.clone().detach()
                gt_draw = mask.clone().detach()
                
                plt.figure(figsize=(12, 6))
                plt.subplot(1, 2, 1)
                plt.imshow(mask_draw.cpu().squeeze(), cmap='gray')
                plt.title('Predicted Mask')
                plt.axis('off')

                plt.subplot(1, 2, 2)
                plt.imshow(gt_draw.cpu().squeeze(), cmap='gray')
                plt.title('Ground Truth Mask')
                plt.axis('off')

                plt.show()

                unique_elements, counts_elements = np.unique(gt_draw.cpu().numpy(), return_counts=True)
                print(f"Unique elements in the mask: {unique_elements}")
                print(f"Counts of elements: {counts_elements}")
            
                # calculate metrics
                iou = compute_iou(mask_pred, mask, include_background=False, ignore_empty=True).mean().item()
                dice = computeDice(mask_pred, mask).item()
                hd = computeHD(mask_pred, mask).item()
                recall = compute_recall(mask_pred_squeezed, mask_squeezed).item()
                precision = compute_precision(mask_pred_squeezed, mask_squeezed).item()

                avgDice.append(dice)
                avgIoU.append(iou)
                avgHD.append(hd)
                avgRec.append(recall)
                avgPrec.append(precision)
                ###################################

                mask_numpy = mask_draw.cpu().detach().numpy()[0][0]
                mask_numpy[mask_numpy == 1] = 255 
                img_id = img_id[0].split('.')[0]
                cv2.imwrite(f'{save_png}{img_id}.png', mask_numpy)
            
                
                torch.cuda.empty_cache()
    
                
    
            result_dict = {'image_id':img_id, 'iou':avgIoU, 'dice':avgDice, 'hd':avgHD, 'recall':avgRec, 'precision':avgPrec}
            result_df = pd.DataFrame(result_dict)
            result_df.to_csv(f'8.13renew_all4_density_results_{args.dataset}.csv',index=False)
            
            print('mean IoU:',round(np.mean(avgIoU),4),round(np.std(avgIoU),4))
            print('mean Recall:',round(np.mean(avgRec),4),round(np.std(avgRec),4))
            print('mean Precision:',round(np.mean(avgPrec),4),round(np.std(avgPrec),4))
            print('mean HD:',round(np.mean(avgHD),4), round(np.std(avgHD),4))
            print('mean Dice:',round(np.mean(avgDice),4),round(np.std(avgDice),4))

