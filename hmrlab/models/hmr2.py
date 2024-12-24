import torch.nn as nn
from .backbones import create_backbone
from .heads import build_smpl_head



class HMR2(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = create_backbone(model_type='vit')
        self.smpl_head = build_smpl_head()
        