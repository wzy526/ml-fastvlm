"""
Qwen2-VL Vision Encoder wrapper for LLaVA-DAT
"""

import torch
import torch.nn as nn
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, CLIPImageProcessor
from PIL import Image
import numpy as np


class Qwen2VLVisionTower(nn.Module):
    """
    Wrapper for Qwen2-VL's vision encoder to make it compatible with LLaVA-DAT framework
    """
    
    def __init__(self, vision_tower_name, args, delay_load=False):
        super().__init__()
        
        self.vision_tower_name = vision_tower_name
        self.is_loaded = False
        self.select_layer = getattr(args, 'mm_vision_select_layer', -1)
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        
        if not delay_load:
            self.load_model()
        else:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(vision_tower_name, trust_remote_code=True)
            if hasattr(config, 'vision_config'):
                self.config = config.vision_config
                self._hidden_size = self.config.hidden_size
                self._num_patches = None
            else:
                raise ValueError(f"Config does not have vision_config: {config}")
    
    def load_model(self, device_map=None):
        if self.is_loaded:
            return
        
        self.full_model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.vision_tower_name,
            torch_dtype=torch.bfloat16,
            device_map=device_map if device_map else 'auto',
            trust_remote_code=True
        )
        
        self.vision_model = self.full_model.visual
        
        self.processor = AutoProcessor.from_pretrained(
            self.vision_tower_name,
            trust_remote_code=True
        )
        self._qwen2vl_processor = self.processor.image_processor
        self.image_processor = self._create_compatible_processor() # (c, h, w)
        
        self.config = self.full_model.config.vision_config
        merger_output_dim = self.vision_model.merger.mlp[-1].out_features
        self._hidden_size = merger_output_dim  # 1536
        
        # 冻结 vision encoder（通常在训练时冻结）
        self.vision_model.requires_grad_(False)
        self.is_loaded = True
    
    def _create_compatible_processor(self):
        try:
            compatible_processor = CLIPImageProcessor.from_pretrained(
                "openai/clip-vit-large-patch14-336"
            )
        except:
            compatible_processor = CLIPImageProcessor(
                size={"shortest_edge": 336},
                crop_size={"height": 336, "width": 336},
                do_center_crop=True,
                do_normalize=True,
                do_resize=True,
                image_mean=[0.48145466, 0.4578275, 0.40821073],
                image_std=[0.26862954, 0.26130258, 0.27577711],
                resample=3, # PIL.Image.BICUBIC
            )
        
        return compatible_processor
    
    @torch.no_grad()
    def forward(self, images, **kwargs):
        """
        前向传播
        
        Args:
            images: 预处理后的图像 tensor [B, C, H, W] 
                    (来自 CLIP processor，标准格式)
        
        Returns:
            image_features: 图像特征 [B, num_patches, hidden_size]
        """
        if not self.is_loaded:
            self.load_model()
        
        if images.ndim != 4:
            raise ValueError(f"Expected 4D image tensor [B, C, H, W], got shape {images.shape}")
        
        B, C, H, W = images.shape
        device = images.device
        dtype = images.dtype

        #[B, C, H, W] → PIL Image → Qwen2-VL processor → [num_patches, channels]
        
        all_pixel_values = []
        all_grid_thw = []
        
        for i in range(B):
            img_tensor = images[i] # pil [3, H, W]
            
            img_tensor_fp32 = img_tensor.float()
            mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], 
                                device=device, dtype=torch.float32).view(3, 1, 1)
            std = torch.tensor([0.26862954, 0.26130258, 0.27577711], 
                               device=device, dtype=torch.float32).view(3, 1, 1)
            img_tensor_fp32 = img_tensor_fp32 * std + mean
            img_tensor_fp32 = torch.clamp(img_tensor_fp32, 0, 1)
            
            # 转换为 numpy [H, W, C]，范围 [0, 255]
            img_np = (img_tensor_fp32.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            pil_image = Image.fromarray(img_np)
            
            processed = self._qwen2vl_processor([pil_image], return_tensors='pt')
            
            # processed['pixel_values']: [num_patches, channels]
            # processed['image_grid_thw']: [1, 3]
            all_pixel_values.append(processed['pixel_values'])
            all_grid_thw.append(processed['image_grid_thw'])
        

        # 每个 pixel_values 是 [num_patches_i, channels]
        pixel_values_batch = torch.cat(all_pixel_values, dim=0).to(device=device, dtype=dtype)
        grid_thw_batch = torch.cat(all_grid_thw, dim=0).to(device=device)
        
        outputs = self.vision_model(
            hidden_states=pixel_values_batch,
            grid_thw=grid_thw_batch
        )
        
        # outputs: [total_patches_merged, 1536]
        
        #重新分割回 [B, num_patches_per_image, hidden_size]
        features_list = []
        offset = 0
        for i in range(B):
            h, w = grid_thw_batch[i, 1].item(), grid_thw_batch[i, 2].item()
            merge_size = self.config.spatial_merge_size
            num_patches_merged = (h // merge_size) * (w // merge_size)
            
            features_list.append(outputs[offset:offset+num_patches_merged])
            offset += num_patches_merged
        
        if len(set([f.shape[0] for f in features_list])) == 1:
            image_features = torch.stack(features_list, dim=0) # [B, num_patches_per_image, hidden_size]
        else:
            raise NotImplementedError("Variable number of patches per image not supported yet")
        
        return image_features
    
    @property
    def hidden_size(self):
        return self._hidden_size
    
    @property
    def num_patches_per_side(self):
        return 12
    
    @property
    def num_patches(self):
        if self._num_patches is None:
            self._num_patches = self.num_patches_per_side ** 2
        return self._num_patches
    
    def to(self, *args, **kwargs):
        if self.is_loaded:
            self.vision_model = self.vision_model.to(*args, **kwargs)
        return super().to(*args, **kwargs)


def build_qwen2vl_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    
    if 'qwen2-vl' in vision_tower.lower() or 'qwen2vl' in vision_tower.lower():
        return Qwen2VLVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    else:
        raise ValueError(f"Not a Qwen2-VL model: {vision_tower}")
