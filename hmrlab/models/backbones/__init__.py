from .vit import ViT

def create_backbone(model_type='vit'):
    if model_type == 'vit':
        return ViT(
                img_size=(256, 192),
                patch_size=16,
                embed_dim=1280,
                depth=32,
                num_heads=16,
                ratio=1,
                use_checkpoint=False,
                mlp_ratio=4,
                qkv_bias=True,
                drop_path_rate=0.55,
            )
        