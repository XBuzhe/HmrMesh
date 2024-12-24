from hmrlab.utils.video_io_utils import read_video_np, get_video_lwh


if __name__ == '__main__':
    video_path = "ldty1_50.mp4"
    L,H,W = get_video_lwh(video_path)
    print(L,H,W)