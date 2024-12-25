import torch
import argparse
import os
import cv2
import torch.nn as nn 
from tqdm import tqdm
from smplx import SMPL
import numpy as np
from hmr2.configs import CACHE_DIR_4DHUMANS
from hmr2.models import HMR2, download_models, load_hmr2, DEFAULT_CHECKPOINT
from hmr2.utils import recursive_to
from hmr2.datasets.vitdet_dataset import ViTDetDataset
from hmr2.utils.renderer import Renderer, cam_crop_to_full
from label_utils.vitpose_utils.get_batch import get_batch
LIGHT_BLUE=(0.65098039,  0.74117647,  0.85882353)
from label_utils.tracker import Tracker
from label_utils.vitpose_utils.get_batch import get_bbx_xys_from_xyxy
from label_utils.vitpose import VitPoseExtractor
from label_utils.vitpose_utils.kp2d_utils import draw_coco17_skeleton_batch
from pytorch3d.transforms import rotation_conversions
from label_utils.vitpose_utils.get_batch import read_video_np

class Loss_kpts2d(nn.Module):
    def __init__(self):
        super(Loss_kpts2d, self).__init__()
        self.loss_fn = nn.MSELoss(reduction='none')
        self.loss_l1 = nn.L1Loss(reduction='none')
        self.threshold = torch.tensor([0.0085, 0.00831, 0.0083, 0.00743, 0.00737, 0.00742, 0.00748, 0.01097, 0.01103, 0.01414, 0.01356, 0.01126, 0.01127, 0.01616, 0.01663, 0.00533, 0.00565])
    def forward(self, pred_kpt, target_kpt):

        kpt2d = pred_kpt
        target_kpt = target_kpt.cuda()
        
        target_kpt2d = target_kpt[...,:2]
        target_kpt2d = target_kpt2d.cuda()
        confidence = target_kpt[..., 2]
        confidence = confidence.unsqueeze(-1).cuda()
        weighted_loss = self.loss_l1(kpt2d, target_kpt2d)*confidence
        weighted_loss = self.loss_l1(kpt2d, target_kpt2d)

        err_kpt = weighted_loss.sum(-1)
        threshold = self.threshold.unsqueeze(0).expand(err_kpt.shape[0], -1).cuda()

        kpts_mask = err_kpt > threshold
        mask_values = torch.where(kpts_mask, torch.tensor(1.0).cuda(), torch.tensor(0.001).cuda())
        loss_kpt2d = err_kpt*mask_values
        loss_kpt2d = loss_kpt2d.mean()
        
        return loss_kpt2d

def fit_loop(model, imgs_batch, targets_kpt2d, loss_fn, optimizer, step, boxes):
    # model.train()
    batch_size = 8
    imgs_batch = imgs_batch.cuda()
    F = imgs_batch.shape[0]
    accumulation_step = F / batch_size
    
    # fiting loop
    pbar = tqdm(range(step), desc="start fitting*", ncols=80)

    for i in pbar:
        sum_loss = 0
        for j in range(0, F, batch_size):
            sub_imgs_batch = imgs_batch[j: j+batch_size]
            sub_label_batch = targets_kpt2d[j: j+batch_size]
            sub_boxes = boxes[j:j+batch_size]
            model_out = model(sub_imgs_batch)
            hmr2_kpts = get_kpts_hmr2(model_out, sub_boxes)

            loss = loss_fn(hmr2_kpts, sub_label_batch)
            
            loss = loss/(accumulation_step+1)
            loss.backward()
            sum_loss = sum_loss + loss
        pbar.set_description(f'Fiting {i}/{step}| loss:{sum_loss}')
        optimizer.step()
        optimizer.zero_grad()

def get_batch_hmr(img_path, bbx):
    imgs, bbx_xys = get_batch(img_path, bbx, img_ds=1,path_type="video")
    return imgs

def get_kpts_hmr2(model_out, boxes):

    smpl2coco  = torch.load('smpl_coco17_J_regressor.pt').unsqueeze(0).cuda()
    pred_cam = model_out['pred_cam']
    F = pred_cam.shape[0]
    box_center = boxes[:,:2].cuda()
    box_size = boxes[:,2].cuda()
    img_size = torch.tensor([[1920.,1080.]]).cuda()
    scaled_focal_length = 5000 / 256 * 1920
    pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length)
    
    smpl = SMPL('data/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl').cuda()
    body_pose = rotation_conversions.matrix_to_axis_angle(model_out['pred_smpl_params']['body_pose']).reshape(F,-1)
    global_orient = rotation_conversions.matrix_to_axis_angle(model_out['pred_smpl_params']['global_orient']).reshape(F,-1)
    smpl_out = smpl(betas=model_out['pred_smpl_params']['betas'], body_pose=body_pose, global_orient=global_orient,transl=pred_cam_t_full)
    verts = smpl_out.vertices
    coco17 = smpl2coco@verts
    z = coco17[...,2].unsqueeze(-1).expand(-1, -1, 3)
    joint = coco17/z
    joint = joint.unsqueeze(-1)
    K = torch.tensor([[scaled_focal_length, 0, 1920/2],[0, scaled_focal_length, 1080/2],[0,0,1]]).unsqueeze(0).unsqueeze(0).cuda()
    coco17_2d = K@joint
    coco17_2d = coco17_2d.squeeze(-1)
    return coco17_2d[...,:2]



def vis_mesh(img, out, focal, cam_t):
    # 可视化hmr2.0推理结果
    smpl = SMPL('data/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl').cuda()
    renderer = Renderer(faces=smpl.faces)

    body_pose = rotation_conversions.matrix_to_axis_angle(out['pred_smpl_params']['body_pose']).reshape(1,-1)
    global_orient = rotation_conversions.matrix_to_axis_angle(out['pred_smpl_params']['global_orient']).reshape(1,-1)
    smpl_out = smpl(betas=out['pred_smpl_params']['betas'], body_pose=body_pose, global_orient=global_orient,transl=cam_t)

    verts = smpl_out.vertices[0]
    smpl2coco = torch.load('smpl_coco17_J_regressor.pt').cuda()
    coco17 = smpl2coco@verts
    coco17 = coco17.cpu()
    joints = smpl_out.joints[0][:24]
    joints = joints.cpu()

    renderer.focal_length = focal.cpu()
    cam_view = renderer.render_rgba(vertices=verts.cpu().numpy(), render_res=[img.shape[1], img.shape[0]])
    # Overlay image
    input_img = img.astype(np.float32)[:,:,::-1]/255.0
    input_img = np.concatenate([input_img, np.ones_like(input_img[:,:,:1])], axis=2) # Add alpha channel
    input_img_overlay = input_img[:,:,:3] * (1-cam_view[:,:,3:]) + cam_view[:,:,:3] * cam_view[:,:,3:]
    mesh_img = 255*input_img_overlay[:,:,::-1]

    # mesh_img = proj_points(mesh_img, joints, focal, (0,0,255))
    mesh_img = proj_points(mesh_img, coco17, focal, (0,0,255))
    return mesh_img
    

def proj_points(img, kpts, focal, color):
    z = kpts[:,2].unsqueeze(-1).expand(-1, 3)
    H,W,_ = img.shape
    joint = kpts/z
    joint = joint.unsqueeze(-1)
    K = torch.tensor([[focal, 0, W/2],[0, focal, H/2],[0,0,1]]).unsqueeze(0)
    zb = K@joint
    # color = (0, 0, 255)
    radius = 3  
    thickness = -1
    for p in zb:
        x = int(p[0].item())
        y = int(p[1].item())
        cv2.circle(img, (x,y), radius, color, thickness)
    return img

def run_preprocess(video_path, verbose):
    video_name = os.path.basename(video_path)[:-4]
    save_path =  os.path.join('label_output', video_name)
    os.makedirs(os.path.join(save_path), exist_ok=True)
    # Get bbx tracking result
    if not os.path.isfile(os.path.join(save_path,'bbx.pt')):
        tracker = Tracker()
        bbx_xyxy = tracker.get_one_track(video_path).float()  # (L, 4)
        torch.save(bbx_xyxy, os.path.join(save_path,'bbx.pt'))
    else:
        bbx_xyxy = torch.load(os.path.join(save_path,'bbx.pt'))
    bbx_xys = get_bbx_xys_from_xyxy(bbx_xyxy, base_enlarge=1.2).float()


    

    # Get VitPose
    if not os.path.isfile(os.path.join(save_path,'kpt.pt')):
        vitpose_extractor = VitPoseExtractor()
        vitpose = vitpose_extractor.extract(video_path, bbx_xys)
        torch.save(vitpose, os.path.join(save_path,'kpt.pt'))
    else:
        vitpose = torch.load(os.path.join(save_path,'kpt.pt'))
    
    bbx_xys_hmr2 = get_bbx_xys_from_xyxy(bbx_xyxy, base_enlarge=1)



    # 可视化
    if verbose:
        from label_utils.get_batch import read_video_np
        from label_utils.kp2d_utils import draw_coco17_skeleton_batch
        from label_utils.kp2d_utils import save_video
        from label_utils.kp2d_utils import draw_bbx_xyxy_on_image_batch
        video = read_video_np(video_path)
        if not os.path.isfile(os.path.join(save_path,'xyxy_overlay.mp4')):
            video_overlay = draw_bbx_xyxy_on_image_batch(bbx_xyxy, video)
            save_video(video_overlay, os.path.join(save_path,'xyxy_overlay.mp4'))

        if not os.path.isfile(os.path.join(save_path,'kpt_overlay.mp4')):
            video_overlay = draw_coco17_skeleton_batch(video, vitpose, 0.5)
            save_video(video_overlay, os.path.join(save_path,'kpt_overlay.mp4'))
    return bbx_xys_hmr2, vitpose


def drow_points(img, kpts):
    kpts = kpts.cpu().int()
    kpts = kpts.squeeze(0).numpy().tolist()
    for i, point in enumerate(kpts):
    # 绘制点，半径为3，颜色为红色 (0, 0, 255)
        cv2.circle(img, point, 2, (0, 255, 0), -1)
    return img

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
    
    smpl = SMPL('data/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl',batch_size=256).cuda()
    smpl2coco = torch.load('smpl_coco17_J_regressor.pt').cuda()
    renderer = Renderer(faces=smpl.faces)
    imgs = read_video_np(video_path, scale=1)[..., ::-1]
    L,H,W,_ = imgs.shape
    batch_size = 24
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
        
        write.write(mesh_img)
def main():
    import time
    parser = argparse.ArgumentParser(description='HMR2 demo code')
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT, help='Path to pretrained model checkpoint')
    parser.add_argument('--video', type=str, default='example_data/videos/ldty1_50.mp4', help='Folder with input images')
    parser.add_argument('--step', type=int, default=30, help='Folder with input images')
    args = parser.parse_args()

    # pre_process and get_label 
    video_path = args.video
    boxes, vitpose = run_preprocess(video_path, verbose=True)
    batch = get_batch_hmr(video_path, boxes)
    # Download and load checkpoints
    download_models(CACHE_DIR_4DHUMANS)
    model, model_cfg = load_hmr2(args.checkpoint)

    # Setup HMR2.0 model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    batch = recursive_to(batch, device)
    model.eval()

    video_name = os.path.basename(video_path)[:-4]
    save_path =  os.path.join('label_output', video_name)
    infer_model_vis(model, batch, video_path, os.path.join(save_path, 'mesh_vis.mp4'), boxes)


    # refine
    loss = Loss_kpts2d()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-6)
    fit_loop(model, batch, vitpose, loss, optimizer, 30, boxes)

    # vis
    infer_model_vis(model, batch, video_path, os.path.join(save_path, 'mesh_vis_refine.mp4'), boxes)


if __name__ == '__main__':
    main()
