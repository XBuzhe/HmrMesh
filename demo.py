import argparse
import torch
import numpy as np
from tqdm import tqdm
from smplx import SMPL
from hmrlab.models.hmr2 import HMR2
from hmrlab.models.tracker import Tracker
from hmrlab.utils import get_bbx_xys_from_xyxy, get_batch
from hmrlab.utils.vis.renderer import Renderer, cam_crop_to_full
from hmrlab.utils.video_io_utils import read_video_np
from hmrlab.utils import rotation_conversions
import cv2

def concat_dict_list(dict_list, dim=0):
    """
    递归拼接字典列表中的字典，字典值为 Tensor，嵌套字典也会被处理。

    Args:
        dict_list (list): 包含字典的列表，每个字典的值是 Tensor。
        dim (int): 拼接的维度，默认是 0。

    Returns:
        dict: 拼接后的字典。
    """
    result = {}

    # 遍历字典列表中的每个字典
    for d in dict_list:
        for key, value in d.items():
            if isinstance(value, torch.Tensor):
                # 如果值是 Tensor，拼接它
                if key not in result:
                    result[key] = value  # 如果还没有该键，初始化为 Tensor，并增加一个维度
                else:
                    result[key] = torch.cat([result[key], value], dim=dim)  # 沿指定维度拼接
            elif isinstance(value, dict):
                # 如果值是嵌套字典，递归处理
                if key not in result:
                    result[key] = concat_dict_list([value], dim=dim)
                else:
                    result[key] = concat_dict_list([result[key], value], dim=dim)

    return result

def infer_model_vis(model, batch, video_path, save_path, boxes):
    
    smpl = SMPL('data/smpl/models/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl',batch_size=256).cuda()
    # smpl2coco = torch.load('smpl_coco17_J_regressor.pt').cuda()
    renderer = Renderer(faces=smpl.faces)
    imgs = read_video_np(video_path)[..., ::-1]
    L,H,W,_ = imgs.shape
    batch_size = 8
    out_list = []
    with torch.no_grad():
        for j in range(0, L, batch_size):
            sub_batch = batch[j:j+batch_size]
            sub_out = model(sub_batch)
            out_list.append(sub_out)
    out = concat_dict_list(out_list)
        
    # kpt_coco_hmr2 = get_kpts_hmr2(out, boxes)
    pred_cam = out['pred_cam']
    box_center = boxes[:,:2].cuda()
    box_size = boxes[:,2].cuda()
    img_size = torch.tensor([[W,H]]).cuda()
    scaled_focal_length = 5000 / 256 * img_size.max()
    renderer.focal_length = scaled_focal_length.cpu()
    pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach()
    body_pose = rotation_conversions.matrix_to_axis_angle(out['pred_smpl_params']['body_pose']).reshape(L,-1)
    global_orient = rotation_conversions.matrix_to_axis_angle(out['pred_smpl_params']['global_orient']).reshape(L,-1)
    smpl_out = smpl(betas=out['pred_smpl_params']['betas'], body_pose=body_pose, global_orient=global_orient,transl=pred_cam_t_full)
    all_verts = smpl_out.vertices
    # 渲染结果
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    write = cv2.VideoWriter(save_path, fourcc, 30, (W, H))

    for i in tqdm(range(L),desc='Render Result...'):
        verts = all_verts[i]
        
        cam_view = renderer.render_rgba(vertices=verts.cpu().numpy(), render_res=[imgs.shape[2], imgs.shape[1]])
        img = imgs[i]
        input_img = img.astype(np.float32)[:,:,::-1]/255.0
        input_img = np.concatenate([input_img, np.ones_like(input_img[:,:,:1])], axis=2)
        input_img_overlay = input_img[:,:,:3] * (1-cam_view[:,:,3:]) + cam_view[:,:,:3] * cam_view[:,:,3:]
        mesh_img = 255*input_img_overlay[:,:,::-1]
        mesh_img = mesh_img.astype(np.uint8)
        # cv2.imwrite('test.png', mesh_img)
        write.write(mesh_img)


def run_preprocess(video_path):
    tracker = Tracker()
    bbx_xyxy = tracker.get_one_track(video_path).float()
    bbx_xys = get_bbx_xys_from_xyxy(bbx_xyxy, base_enlarge=1.0).float()
    imgs, _ = get_batch(video_path, bbx_xys, img_ds=1)
    return imgs, bbx_xys

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HMR2 demo code')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/hmr2/epoch=35-step=1000000.ckpt', help='Path to pretrained model checkpoint')
    parser.add_argument('--video', type=str, default='test.mp4', help='Path to pretrained model checkpoint')

    args = parser.parse_args()

    model = HMR2()
    model.load_state_dict(torch.load('checkpoint\hmr2\hmr2b_b.pt'),strict=False)
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    model.eval()

    video = args.video
    imgs, bbx_xys = run_preprocess(video)
    
    infer_model_vis(model, imgs.cuda(), video, 'out.mp4', bbx_xys)


