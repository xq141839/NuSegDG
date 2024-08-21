import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch import optim
import time
from torch.optim import lr_scheduler
import seaborn as sns
import pandas as pd
import argparse
import os
from dataloader import BinaryLoader
from loss import *
from tqdm import tqdm
import json
from model import SAMB
from functools import partial
import albumentations as A
from albumentations.pytorch.transforms import ToTensor

torch.set_num_threads(8)
# matplotlib.use('TkAgg')

def train_model(model, criterion_mask, optimizer, scheduler, num_epochs=5):
    since = time.time()
    
    Loss_list = {'train': [], 'valid': []}
    Accuracy_list = {'train': [], 'valid': []}
    
    best_model_wts = model.state_dict()

    best_loss = float('inf')
    counter = 0
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train(True)
    
            else:
                model.train(False)  

            running_loss_mask = []
            running_corrects_mask = []
        
            # Iterate over data
            #for inputs,labels,label_for_ce,image_id in dataloaders[phase]: 
            for _, img, labels, img_id, density_map in tqdm(dataloaders[phase]):      
                # wrap them in Variable
    
                img = Variable(img.cuda())
                labels = Variable(labels.cuda())
                density_map = Variable(density_map.cuda())

                # zero the parameter gradients
                optimizer.zero_grad()
             
                pred_mask = model(x=img, mask=density_map, img_id=img_id)
                pred_mask = torch.sigmoid(pred_mask)
                

                # ensure pred_mask and labels are Float type
                pred_mask = pred_mask.float()
                labels = labels.float()
                
                loss = criterion_mask(pred_mask, labels)
                score_mask1 = accuracy_metric(pred_mask, labels)
    

                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    
                # calculate loss and IoU
                running_loss_mask.append(loss.item())
                running_corrects_mask.append(score_mask1.item())
             

            epoch_loss = np.mean(running_loss_mask)
            epoch_acc = np.mean(running_corrects_mask)
            
            print('{} Loss: {:.4f} IoU: {:.4f} '.format(
                phase, np.mean(running_loss_mask), np.mean(running_corrects_mask)))
            
            Loss_list[phase].append(epoch_loss)
            Accuracy_list[phase].append(epoch_acc)

            # save parameters
            if phase == 'valid' and epoch_loss <= best_loss:
                best_loss = epoch_loss
                best_model_wts = model.state_dict()
                torch.save(best_model_wts, f'outputs/8.13Renew_{args.dataset}_epoch_{epoch}.pth')
            if phase == 'valid':
                scheduler.step()
        
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))
    

    return Loss_list, Accuracy_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,default='all', help='monuseg2018, dsb2018, SegPC, CryoNuSeg, TNBC')
    parser.add_argument('--sam_pretrain', type=str,default='../pretrain/sam_vit_b_01ec64.pth', 
    help='pretrain/sam_vit_b_01ec64.pth, medsam_box_best_vitb.pth, medsam_vit_b')
    parser.add_argument('--jsonfile', type=str,default='data_split.json', help='')
    parser.add_argument('--batch', type=int, default=2, help='batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--epoch', type=int, default=50, help='epoches')
    args = parser.parse_args()

    os.makedirs('outputs/', exist_ok=True)

    args.jsonfile = f'datasets/{args.dataset}/data_split.json'
    
    with open(args.jsonfile, 'r') as f:
        df = json.load(f)
    
    val_files = df['valid']
    train_files = df['train']  
    
    train_dataset = BinaryLoader(args.dataset, train_files, A.Compose([
        A.Resize(1024, 1024),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensor()
        ], 
        additional_targets={'mask2': 'mask'}))
    val_dataset = BinaryLoader(args.dataset, val_files, A.Compose([
        A.Resize(1024, 1024),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensor()
        ],
        additional_targets={'mask2': 'mask'}))
    
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch, shuffle=True,drop_last=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=1)
    
    dataloaders = {'train':train_loader,'valid':val_loader}
    
    model = SAMB(data_path=f'datasets/{args.dataset}') #SAMB

    encoder_dict = torch.load(args.sam_pretrain)
    pre_dict = {k: v for k, v in encoder_dict.items() if list(k.split('.'))[0] == 'image_encoder'}
    model.load_state_dict(pre_dict, strict=False)

    pre_dict = {k: v for k, v in encoder_dict.items() if list(k.split('.'))[0] == 'prompt_encoder'}
    model.load_state_dict(pre_dict, strict=False)

    pre_dict = {k: v for k, v in encoder_dict.items() if list(k.split('.'))[0] == 'mask_decoder'}
    model.load_state_dict(pre_dict, strict=False)

    # model.load_state_dict(torch.load(args.sam_pretrain), strict=True)
    # model.load_state_dict(torch.load(args.sam_pretrain, map_location={'cuda:4': 'cuda:0', 'cuda:6': 'cuda:0'}), strict=True)
    # model.load_state_dict(torch.load(args.sam_pretrain)["model"], strict=True)


    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")

        model = nn.DataParallel(model)

    model = model.cuda()

    # for n, value in model.prompt_encoder.named_parameters():
    #     value.requires_grad = False

    for n, value in model.module.image_encoder.named_parameters():
        if f"train" in n:
            value.requires_grad = True
        else:
            value.requires_grad = False

    trainable_params = sum(
	p.numel() for p in model.parameters() if p.requires_grad
    )

    print('Trainable Params = ' + str(trainable_params/1000**2) + 'M')

    total_params = sum(
	param.numel() for param in model.parameters()
    )

    print('Total Params = ' + str(total_params/1000**2) + 'M')

    print('Ratio = ' + str(trainable_params/total_params) + '%')

        
    # Loss, IoU and Optimizer
    mask_loss = BinaryMaskLoss() # nn.CrossEntropyLoss()
    accuracy_metric = BinaryIoU()#BinaryIoU()

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),lr = args.lr)
    # optimizer = optim.Adam(model.parameters(),lr = args.lr)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.8)
    # exp_lr_scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.95, verbose=True)
    # exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5,min_lr=1e-7)
    Loss_list, Accuracy_list = train_model(model, mask_loss, optimizer, exp_lr_scheduler,
                           num_epochs=args.epoch)
    
    plt.title('Validation loss and IoU',)
    valid_data = pd.DataFrame({'Loss':Loss_list["valid"], 'IoU':Accuracy_list["valid"]})
    valid_data.to_csv(f'valid_data.csv')
    sns.lineplot(data=valid_data,dashes=False)
    plt.ylabel('Value')
    plt.xlabel('Epochs')
    plt.savefig('valid.png')
    
    plt.figure()
    plt.title('Training loss and IoU',)
    valid_data = pd.DataFrame({'Loss':Loss_list["train"],'IoU':Accuracy_list["train"]})
    valid_data.to_csv(f'train_data.csv')
    sns.lineplot(data=valid_data,dashes=False)
    plt.ylabel('Value')
    plt.xlabel('Epochs')
    plt.savefig('train.png')
