import numpy as np
import torch
from tqdm import tqdm
import cv2


def get_video_lwh(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Error: Could not open video.")
    return int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))


def read_video_np(video_path):
    """
    Args:
        video_path: str
    Returns:
        frames: np.array, (N, H, W, 3) RGB, uint8
    """
    # If video path not exists, an error will be raised by ffmpegs
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Error: Could not open video.")
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        frames_array = np.array(frames)
    cap.release()
    return frames_array


def get_video_reader(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Error: Could not open video {video_path}.")
    return cap


def read_images_np(image_paths, verbose=False):
    """
    Args:
        image_paths: list of str
    Returns:
        images: np.array, (N, H, W, 3) RGB, uint8
    """
    if verbose:
        images = [cv2.imread(str(img_path))[..., ::-1] for img_path in tqdm(image_paths)]
    else:
        images = [cv2.imread(str(img_path))[..., ::-1] for img_path in image_paths]
    images = np.stack(images, axis=0)
    return images


def save_video(images, video_path, fps=30):
    
    if isinstance(images, torch.Tensor):
        images = images.cpu().numpy().astype(np.uint8)
    elif isinstance(images, list):
        images = np.array(images).astype(np.uint8)

    height, width = images.shape[1], images.shape[2]

    fourcc = cv2.VideoWriter_fourcc(*'X264')  
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    for image in images:
        video_writer.write(image)

    video_writer.release()
