import os
from .clip_encoder import CLIPVisionTower, CLIPVisionTowerS2, CLIPVisionTowerLLaVA16
from .mobileclip_encoder import MobileCLIPVisionTower


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    is_absolute_path_exists = os.path.exists(vision_tower)
    use_s2 = getattr(vision_tower_cfg, 's2', False)
    use_llava16 = getattr(vision_tower_cfg, 'llava16', False)

    if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion") or "ShareGPT4V" in vision_tower:
        if use_llava16:
            return CLIPVisionTowerLLaVA16(vision_tower, args=vision_tower_cfg, **kwargs)
        elif use_s2:
            return CLIPVisionTowerS2(vision_tower, args=vision_tower_cfg, **kwargs)
        else:
            return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    elif "mobileclip" in vision_tower.lower():
        return MobileCLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}')
