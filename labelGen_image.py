import torch
import argparse
import os
import cv2
import torch.nn as nn 
from smplx import SMPL
import numpy as np
from hmr2.configs import CACHE_DIR_4DHUMANS
from hmr2.models import HMR2, download_models, load_hmr2, DEFAULT_CHECKPOINT
from hmr2.utils import recursive_to
from hmr2.datasets.vitdet_dataset import ViTDetDataset
from hmr2.utils.renderer import Renderer, cam_crop_to_full
from label_utils.vitpose_utils.get_batch import get_batch
LIGHT_BLUE=(0.65098039,  0.74117647,  0.85882353)

from label_utils.vitpose_utils.get_batch import get_bbx_xys_from_xyxy
from label_utils.vitpose import VitPoseExtractor
from label_utils.vitpose_utils.kp2d_utils import draw_coco17_skeleton_batch
from pytorch3d.transforms import rotation_conversions


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
    batch_size = 16
    imgs_batch = imgs_batch.cuda()
    F = imgs_batch.shape[0]
    # accumulation_step = F / batch_size
    from tqdm import tqdm
    # fiting loop
    pbar = tqdm(range(step), desc="start fitting*", ncols=80)

    for i in pbar:
        model_out = model(imgs_batch)
        hmr2_kpts = get_kpts_hmr2(model_out, boxes)
        loss = loss_fn(hmr2_kpts, targets_kpt2d)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        pbar.set_description(f'Fiting {i}/{step}| loss:{loss}')

def get_batch_hmr(img_path, bbx):
    imgs, bbx_xys = get_batch(img_path, bbx, img_ds=1,path_type="image")
    return imgs

def get_kpts_hmr2(model_out, boxes):

    smpl2coco  = torch.load('smpl_coco17_J_regressor.pt').unsqueeze(0).cuda()
    pred_cam = model_out['pred_cam']
    box_center = boxes[:,:2].cuda()
    box_size = boxes[:,2].cuda()
    img_size = torch.tensor([[1920.,1080.]]).cuda()
    scaled_focal_length = 5000 / 256 * 1920
    pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length)
    
    smpl = SMPL('data/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl').cuda()
    body_pose = rotation_conversions.matrix_to_axis_angle(model_out['pred_smpl_params']['body_pose']).reshape(1,-1)
    global_orient = rotation_conversions.matrix_to_axis_angle(model_out['pred_smpl_params']['global_orient']).reshape(1,-1)
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

def run_preprocess(img_path, verbose):
    img_name = os.path.basename(img_path)[:-4]
    save_path =  os.path.join('label_output', img_name)
    os.makedirs(os.path.join(save_path), exist_ok=True)
    # Get bbx tracking result
    from ultralytics import YOLO
    yolo = YOLO("/gemini/data-1/2codehmr/GVHMR/inputs/checkpoints/yolo/yolov8x.pt")
    result = yolo(img_path, classes=[0])[0]
    # 计算主体人物的bbox
    boxes = result.boxes.xyxy.cpu()
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    areas = widths * heights
    max_area_index = torch.argmax(areas)
    bbx_xyxy = boxes[max_area_index]
    bbx_xyxy = bbx_xyxy[None]

    bbx_xys = get_bbx_xys_from_xyxy(bbx_xyxy, base_enlarge=1.2).float()

    # Get VitPose
    if not os.path.isfile(os.path.join(save_path,'kpt.pt')):
        vitpose_extractor = VitPoseExtractor()
        vitpose = vitpose_extractor.extract(img_path, bbx_xys)
        torch.save(vitpose, os.path.join(save_path,'kpt.pt'))
    else:
        vitpose = torch.load(os.path.join(save_path,'kpt.pt'))

    bbx_xys_hmr2 = get_bbx_xys_from_xyxy(bbx_xyxy, base_enlarge=1)



    # 可视化
    if verbose:
        img = cv2.imread(img_path)
        img = cv2.rectangle(img, (int(bbx_xyxy[0][0]),int(bbx_xyxy[0][1])), (int(bbx_xyxy[0][2]),int(bbx_xyxy[0][3])), (0, 255, 0), 2)
        img = draw_coco17_skeleton_batch(img[None], vitpose, 0.5)
        cv2.imwrite(os.path.join(save_path,os.path.basename(img_path)), img[0])

    return bbx_xys_hmr2, vitpose



def main():
    import time
    start = time.time()
    parser = argparse.ArgumentParser(description='HMR2 demo code')
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT, help='Path to pretrained model checkpoint')
    parser.add_argument('--img', type=str, default='first_frame.jpg', help='Folder with input images')
    parser.add_argument('--out_folder', type=str, default='demo_out', help='Output folder to save rendered results')
    parser.add_argument('--side_view', dest='side_view', action='store_true', default=False, help='If set, render side view also')
    parser.add_argument('--top_view', dest='top_view', action='store_true', default=False, help='If set, render top view also')
    parser.add_argument('--full_frame', dest='full_frame', action='store_true', default=False, help='If set, render all people together also')
    parser.add_argument('--save_mesh', dest='save_mesh', action='store_true', default=False, help='If set, save meshes to disk also')
    parser.add_argument('--detector', type=str, default='vitdet', choices=['vitdet', 'regnety'], help='Using regnety improves runtime')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for inference/fitting')
    parser.add_argument('--file_type', nargs='+', default=['*.jpg', '*.png'], help='List of file extensions to consider')

    args = parser.parse_args()

    # Download and load checkpoints
    download_models(CACHE_DIR_4DHUMANS)
    model, model_cfg = load_hmr2(args.checkpoint)

    # Setup HMR2.0 model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    model.eval()


    img_path = args.img
    img_name = os.path.basename(img_path)[:-4]
    save_path =  os.path.join('label_output', img_name)

    img_cv2 = cv2.imread(str(img_path))

    # Detect humans in image use yolo
    boxes, vitpose = run_preprocess(img_path, verbose=True)
    
    # Run HMR2.0 on all detected humans
   
    # for batch in dataloader:
    batch = get_batch_hmr(img_path, boxes)
    batch = recursive_to(batch, device)

    # vis result
    with torch.no_grad():
        out = model(batch)

    pred_cam = out['pred_cam']
    box_center = boxes[:,:2].cuda()
    box_size = boxes[:,2].cuda()
    img_size = torch.tensor([[1920.,1080.]]).cuda()
    scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
    pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach()
    overlay_img = vis_mesh(img_cv2.copy(), out, scaled_focal_length, pred_cam_t_full)
    cv2.imwrite(os.path.join(save_path, 'ori.png'), overlay_img)

    loss = Loss_kpts2d()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-6)
    fit_loop(model, batch, vitpose, loss, optimizer, 30, boxes)

    model.eval()
    with torch.no_grad():
        out = model(batch)

    pred_cam = out['pred_cam']
    box_center = boxes[:,:2].cuda()
    box_size = boxes[:,2].cuda()
    img_size = torch.tensor([[1920.,1080.]]).cuda()
    scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
    pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach()
    overlay_img = vis_mesh(img_cv2.copy(), out, scaled_focal_length, pred_cam_t_full)
    cv2.imwrite(os.path.join(save_path, 'refine.png'), overlay_img)


if __name__ == '__main__':
    main()
