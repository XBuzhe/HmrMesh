from .hmr2_head import SMPLTransformerDecoderHead

def build_smpl_head(cfg):
    return SMPLTransformerDecoderHead(cfg)
