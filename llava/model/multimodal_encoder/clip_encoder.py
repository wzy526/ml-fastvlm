import torch
import torch.nn as nn

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig


class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        self.tune_vision_tower = getattr(args, 'unfreeze_mm_vision_tower', False)
        self.input_image_size = getattr(args, 'input_image_size', None)

        if self.tune_vision_tower:
            print("CLIP Vision tower is set to tunable")

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)
            if self.input_image_size is not None:
                self.cfg_only.image_size = self.input_image_size

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        if not self.tune_vision_tower:
            self.vision_tower.requires_grad_(False)

        if self.input_image_size is not None:
            print("Using input image size: {}".format(self.input_image_size))
            self.image_processor.size['shortest_edge'] = self.input_image_size
            self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.input_image_size

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    def forward(self, images):
        if self.tune_vision_tower:
            return self.forward_images(images)
        else:
            with torch.no_grad():
                return self.forward_images(images)

    def forward_images(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2



class CLIPVisionTowerS2(CLIPVisionTower):
    def __init__(self, vision_tower, args, delay_load=False):
        self.s2_scales = getattr(args, 's2_scales', '336,672,1008')
        self.s2_scales = list(map(int, self.s2_scales.split(',')))
        self.s2_scales.sort()
        self.s2_split_size = self.s2_scales[0]
        self.s2_image_size = self.s2_scales[-1]

        super().__init__(vision_tower, args, delay_load)

        try:
            from s2wrapper import forward as multiscale_forward
        except ImportError:
            raise ImportError('Package s2wrapper not found! Please install by running: \npip install git+https://github.com/bfshi/scaling_on_scales.git')
        self.multiscale_forward = multiscale_forward

        # change resize/crop size in preprocessing to the largest image size in s2_scale
        if not delay_load or getattr(args, 'unfreeze_mm_vision_tower', False):
            self.image_processor.size['shortest_edge'] = self.s2_image_size
            self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.s2_image_size

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)

        self.image_processor.size['shortest_edge'] = self.s2_image_size
        self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.s2_image_size

        self.is_loaded = True

    @torch.no_grad()
    def forward_feature(self, images):
        image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
        image_features = self.feature_select(image_forward_outs).to(images.dtype)
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_feature = self.multiscale_forward(self.forward_feature, image.unsqueeze(0), img_sizes=self.s2_scales, max_split_size=self.s2_split_size)
                image_features.append(image_feature)
        else:
            image_features = self.multiscale_forward(self.forward_feature, images, img_sizes=self.s2_scales, max_split_size=self.s2_split_size)

        return image_features

    @property
    def hidden_size(self):
        return self.config.hidden_size * len(self.s2_scales)


class CLIPVisionTowerLLaVA16(CLIPVisionTower):
    """
    CLIP Vision Tower with LLaVA-1.6 style patching logic
    Implements the same patching strategy as LLaVA-1.6 for high-resolution image processing
    """
    
    def __init__(self, vision_tower, args, delay_load=False):
        # LLaVA-1.6 style configuration
        self.llava16_grid_pinpoints = getattr(args, 'llava16_grid_pinpoints', '[336, 672, 1008]')
        self.llava16_patch_size = getattr(args, 'llava16_patch_size', 336)
        self.llava16_image_size = getattr(args, 'llava16_image_size', 1008)
        
        # Parse grid pinpoints
        if isinstance(self.llava16_grid_pinpoints, str):
            import ast
            self.llava16_grid_pinpoints = ast.literal_eval(self.llava16_grid_pinpoints)
        
        super().__init__(vision_tower, args, delay_load)
        
        # Import LLaVA-1.6 utilities
        from llava.mm_utils import process_anyres_image, get_anyres_image_grid_shape
        self.process_anyres_image = process_anyres_image
        self.get_anyres_image_grid_shape = get_anyres_image_grid_shape
        
        # Update image processor settings
        if not delay_load or getattr(args, 'unfreeze_mm_vision_tower', False):
            self.image_processor.size['shortest_edge'] = self.llava16_image_size
            self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.llava16_image_size

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        if not self.tune_vision_tower:
            self.vision_tower.requires_grad_(False)

        # Update processor settings for LLaVA-1.6 style processing
        self.image_processor.size['shortest_edge'] = self.llava16_image_size
        self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.llava16_image_size

        self.is_loaded = True

    @torch.no_grad()
    def forward_feature(self, images):
        """Standard CLIP forward for individual patches"""
        image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
        image_features = self.feature_select(image_forward_outs).to(images.dtype)
        return image_features

    @torch.no_grad()
    def forward(self, images):
        """
        LLaVA-1.6 style forward with patching logic
        Processes high-resolution images by dividing them into patches
        """
        if type(images) is list:
            image_features = []
            for image in images:
                # Process single image with LLaVA-1.6 style patching
                image_feature = self._process_single_image_llava16(image)
                image_features.append(image_feature)
        else:
            # Process batch of images
            if images.ndim == 4:  # Single image in batch
                image_features = self._process_single_image_llava16(images)
            else:  # Multiple images
                image_features = []
                for i in range(images.shape[0]):
                    image_feature = self._process_single_image_llava16(images[i])
                    image_features.append(image_feature)

        return image_features

    def _process_single_image_llava16(self, image):
        """
        Process a single image using LLaVA-1.6 style patching
        """
        # Convert tensor to PIL if needed
        if isinstance(image, torch.Tensor):
            from PIL import Image
            import torchvision.transforms as transforms
            
            # Convert tensor to PIL image
            if image.ndim == 3:
                image = image.unsqueeze(0)
            
            # Assuming image is in format [C, H, W] or [B, C, H, W]
            if image.shape[1] == 3:  # [B, C, H, W]
                image = image.squeeze(0)  # Remove batch dimension
            
            # Convert to PIL
            to_pil = transforms.ToPILImage()
            image = to_pil(image)

        image_patches = self.process_anyres_image(
            image, 
            self.image_processor, 
            self.llava16_grid_pinpoints
        )
        
        # Forward through CLIP for each patch
        patch_features = []
        for patch in image_patches:
            if patch.ndim == 3:
                patch = patch.unsqueeze(0)
            patch_feature = self.forward_feature(patch)
            patch_features.append(patch_feature)
        
        # Concatenate all patch features
        # First patch is the original resized image, rest are patches
        if patch_features:
            # Concatenate along the patch dimension
            image_features = torch.cat(patch_features, dim=0)
            return image_features
        else:
            return torch.empty(0, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def hidden_size(self):
        """Return hidden size for LLaVA-1.6 style processing"""
        # For LLaVA-1.6, we need to account for multiple patches
        # Base hidden size plus patches
        base_size = self.config.hidden_size
        # Number of patches depends on the grid pinpoints configuration
        # This is a simplified calculation
        return base_size

    @property
    def num_patches_per_side(self):
        """Number of patches per side for LLaVA-1.6 style processing"""
        return self.llava16_image_size // self.llava16_patch_size

    @property
    def num_patches(self):
        """Total number of patches for LLaVA-1.6 style processing"""
        return (self.llava16_image_size // self.llava16_patch_size) ** 2
