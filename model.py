import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch
import cv2
import os
from SAM.modeling.mask_decoder import MaskDecoder
from SAM.modeling.prompt_encoder import PromptEncoder
from SAM.modeling.transformer import TwoWayTransformer
from SAM.modeling.common import LayerNorm2d
from SAM.modeling.image_encoder import ImageEncoderViT
from SAM.modeling.small_encoder import TinyViT 
from functools import partial
from transforms import ResizeLongestSide
from typing import Any, Dict, List, Tuple

def PointGenerator(mask, visual=False):
    

    print(f'PointGenerator - TYPE: {type(mask)}, SHAPE: {mask.shape}, TYPE: {mask.dtype}')
    
    if not isinstance(mask, np.ndarray):
        raise ValueError("mask NOT NumPy ARRAY")
            
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
        
    if mask.ndim > 2:
        mask = mask.squeeze()

        
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    point_coord = []
    point_class = []
    for i in range(1, num_labels):
        x, y = centroids[i]
        point_coord.append([x, y])
        point_class.append(1)
    if visual:
        for point in point_coord:
            cv2.circle(mask, (int(point[0]), int(point[1])), 5, (255, 0, 0), -1)
    box_coord = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        box_coord.append([x, y, x+w, y+h])
    mask_shape = mask.shape
    return point_coord, point_class, mask_shape, box_coord



class SAMB(nn.Module):

    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(self, data_path=None, img_size=1024, pixel_mean=[123.675, 116.28, 103.53],pixel_std=[58.395, 57.12, 57.375]):
        super(SAMB, self).__init__()

        ###################################
        self.image_encoder = ImageEncoderViT(
            depth=12,
            embed_dim=768,
            img_size=1024,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=12,
            patch_size=16,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=[2, 5, 8, 11],
            window_size=14,
            out_chans=256,
        )


        self.prompt_encoder = PromptEncoder(
            embed_dim=256,
            image_embedding_size=(1024 // 16, 1024 // 16),
            input_image_size=(1024, 1024),
            mask_in_chans=16,
        )
        
        self.mask_decoder = MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=256,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=256,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        )

        ###################################
        self.path = data_path
        self.img_size = img_size
        self.pt = ResizeLongestSide(img_size)
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

    @property
    def device(self) -> Any:
        return self.pixel_mean.device
    
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x
    
    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def forward(self, x, mask=None, img_id=None, mode='point'):

        b = x.shape[0]

        image_embeddings = self.image_encoder(x)

        outputs_mask = []
        
        for idx in range(b): # for each batch 

            ###### get point and box
            ######

            # get point and box
            point_coord, point_class, mask_shape, box_coord = PointGenerator(mask[idx].cpu().numpy())

            
            if mode == 'point':
                density_map = mask[idx].unsqueeze(0).float()
                if density_map.dim() == 2:
                    density_map = density_map.unsqueeze(0).unsqueeze(0)  # ??[H, W]¡À???[1, 1, H, W]
                elif density_map.dim() == 3:
                    density_map = density_map.unsqueeze(1)

                print("density_map shape:", density_map.shape)
                point_coord = torch.tensor(point_coord, device=self.device).float().unsqueeze(0)
                point_class = torch.tensor(point_class, device=self.device).long().unsqueeze(0)

                sparse_embeddings, dense_embeddings = self.prompt_encoder(
                    points=(point_coord, point_class),
                    #points=None,
                    boxes=None,
                    masks = density_map,
                )
                
                

            elif mode == 'box':
                box_coord = torch.tensor(box_coord, device=x.device).float()
                sparse_embeddings, dense_embeddings = self.prompt_encoder(
                    points=None,
                    masks=None,
                    boxes=box_coord
                )

                            
                
            low_res_masks = self.mask_decoder(
                image_embeddings=image_embeddings[idx].unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )

            masks = F.interpolate(low_res_masks[0], (self.img_size, self.img_size), mode="bilinear", align_corners=False)
            outputs_mask.append(masks.squeeze(0))
            
        return torch.stack(outputs_mask, dim=0)

 

