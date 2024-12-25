import argparse
import torch
from hmrlab.models.hmr2 import HMR2



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HMR2 demo code')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/hmr2/epoch=35-step=1000000.ckpt', help='Path to pretrained model checkpoint')
    args = parser.parse_args()
    model = HMR2()
    model.load_state_dict(torch.load('checkpoints/hmr2/hmr2_b_b.pt'))
    
    data = torch.rand(1,3,256,256)
    with torch.no_grad():
        out = model(data)
        print(0)