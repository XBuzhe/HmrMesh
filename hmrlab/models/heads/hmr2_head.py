import torch
import torch.nn as nn
import numpy as np
import einops

from ...utils.geometry import rot6d_to_rotmat, aa_to_rotmat
from ...networks.pose_transformer import TransformerDecoder


class SMPLTransformerDecoderHead(nn.Module):
    """ Cross-attention based SMPL Transformer decoder
    """

    def __init__(self):
        super().__init__()
        self.joint_rep_type = '6d'
        self.joint_rep_dim = {'6d': 6, 'aa': 3}[self.joint_rep_type]
        npose = self.joint_rep_dim * 24
        self.npose = npose
        self.input_is_mean_shape = False

        transformer_args = dict(
            num_tokens=1,
            token_dim=(npose + 10 + 3) if self.input_is_mean_shape else 1,
            dim=1024,
            depth=6,
            heads=8,
            mlp_dim=1024,
            dim_head=64,
            dropout=0.0,
            emb_dropout=0.0,
            norm='layer',
            context_dim=1280
        )

        self.transformer = TransformerDecoder(
            **transformer_args
        )
        dim=transformer_args['dim']
        self.decpose = nn.Linear(dim, npose)
        self.decshape = nn.Linear(dim, 10)
        self.deccam = nn.Linear(dim, 3)

        mean_params = np.load('data/hmr2/smpl_mean_params.npz')
        init_body_pose = torch.from_numpy(mean_params['pose'].astype(np.float32)).unsqueeze(0)
        init_betas = torch.from_numpy(mean_params['shape'].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam'].astype(np.float32)).unsqueeze(0)
        self.register_buffer('init_body_pose', init_body_pose)
        self.register_buffer('init_betas', init_betas)
        self.register_buffer('init_cam', init_cam)

    def forward(self, x, **kwargs):

        batch_size = x.shape[0]
        # vit pretrained backbone is channel-first. Change to token-first
        x = einops.rearrange(x, 'b c h w -> b (h w) c')

        init_body_pose = self.init_body_pose.expand(batch_size, -1)
        init_betas = self.init_betas.expand(batch_size, -1)
        init_cam = self.init_cam.expand(batch_size, -1)

        # TODO: Convert init_body_pose to aa rep if needed
        if self.joint_rep_type == 'aa':
            raise NotImplementedError

        pred_body_pose = init_body_pose
        pred_betas = init_betas
        pred_cam = init_cam
        pred_body_pose_list = []
        pred_betas_list = []
        pred_cam_list = []
        for i in range(1):
            # Input token to transformer is zero token
            if self.input_is_mean_shape:
                token = torch.cat([pred_body_pose, pred_betas, pred_cam], dim=1)[:,None,:]
            else:
                token = torch.zeros(batch_size, 1, 1).to(x.device)

            # Pass through transformer
            token_out = self.transformer(token, context=x)
            token_out = token_out.squeeze(1) # (B, C)

            # Readout from token_out
            pred_body_pose = self.decpose(token_out) + pred_body_pose
            pred_betas = self.decshape(token_out) + pred_betas
            pred_cam = self.deccam(token_out) + pred_cam
            pred_body_pose_list.append(pred_body_pose)
            pred_betas_list.append(pred_betas)
            pred_cam_list.append(pred_cam)

        # Convert self.joint_rep_type -> rotmat
        joint_conversion_fn = {
            '6d': rot6d_to_rotmat,
            'aa': lambda x: aa_to_rotmat(x.view(-1, 3).contiguous())
        }[self.joint_rep_type]

        pred_smpl_params_list = {}
        pred_smpl_params_list['body_pose'] = torch.cat([joint_conversion_fn(pbp).view(batch_size, -1, 3, 3)[:, 1:, :, :] for pbp in pred_body_pose_list], dim=0)
        pred_smpl_params_list['betas'] = torch.cat(pred_betas_list, dim=0)
        pred_smpl_params_list['cam'] = torch.cat(pred_cam_list, dim=0)
        pred_body_pose = joint_conversion_fn(pred_body_pose).view(batch_size, 24, 3, 3)

        pred_smpl_params = {'global_orient': pred_body_pose[:, [0]],
                            'body_pose': pred_body_pose[:, 1:],
                            'betas': pred_betas}
        return pred_smpl_params, pred_cam, pred_smpl_params_list
