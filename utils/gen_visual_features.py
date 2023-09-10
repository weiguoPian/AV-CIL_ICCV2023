import os

# Modify 3 to your avaliable gpu id
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import sys

import torch
from transformers import VideoMAEImageProcessor, VideoMAEModel

import numpy as np

from PIL import Image
import argparse
import h5py


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='AVE', choices=['AVE', 'ksounds', 'VGGSound_100'])

args = parser.parse_args()

ImageProcessor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")

model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")

model = model.cuda()

model.eval()

frames_root = '../raw_data/{}/frames/'.format(args.dataset)

vid_name_list = os.listdir(frames_root)

input_num_frames = 16

visual_pretrained_feature_dict = {}

if args.dataset == 'AVE':
    save_path = '../data/AVE/visual_pretrained_feature/visual_pretrained_feature_dict.npy'
else:
    save_path = '../data/{}/visual_pretrained_feature/visual_features.h5'.format(args.dataset)
    f = h5py.File(save_path, 'w')


n = 0
for name in vid_name_list:
    print(n, name)
    v_frames_dir = os.path.join(frames_root, name)
    total_frames_num = len(os.listdir(v_frames_dir))

    interval = total_frames_num / (input_num_frames - 1)

    visual_frames = []
    visual_frames.append(Image.open(os.path.join(v_frames_dir, str(1).zfill(6) + '.jpg')))

    for i in range(input_num_frames - 2):
        idx = 1 + int((i + 1) * interval + 0.5)
        visual_frames.append(Image.open(os.path.join(v_frames_dir, str(idx).zfill(6) + '.jpg')))
    visual_frames.append(Image.open(os.path.join(v_frames_dir, str(total_frames_num).zfill(6) + '.jpg')))

    inputs = ImageProcessor(visual_frames, return_tensors='pt')
    inputs = inputs['pixel_values'].cuda()

    with torch.no_grad():
        feature = model(inputs).last_hidden_state
    feature = feature.squeeze(dim=0).detach().cpu().numpy()

    if args.dataset == 'AVE':
        visual_pretrained_feature_dict[name] = feature
    else:
        d = f.create_dataset(name, data=feature)

    n += 1

if args.dataset == 'AVE':
    np.save(save_path, visual_pretrained_feature_dict)
else:
    f.close()



