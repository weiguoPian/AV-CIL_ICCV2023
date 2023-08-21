import os
import numpy as np
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='AVE', choices=['AVE', 'ksounds', 'VGGSound_100'])

args = parser.parse_args()



def extract_frames(video, dst):
    cmd1 = 'ffmpeg '
    cmd1 += '-i ' + video + " "
    cmd1 += '-y' + " "
    if args.dataset == 'AVE':
        cmd1 += '-r ' + "10 "
    else:
        cmd1 += '-r ' + "8 "
    cmd1 += '{0}/%06d.jpg'.format(dst)

    print(cmd1)
    os.system(cmd1)


video_root = '../raw_data/{}/videos/'.format(args.dataset)

frame_root = '../raw_data/{}/frames/'.format(args.dataset)


if not os.path.exists(frame_root):
    os.makedirs(frame_root)

v_name_list = os.listdir(video_root)
for v_name in v_name_list:
    v_id = v_name[:-4]
    v_path = os.path.join(video_root, v_name)
    dst_path = os.path.join(frame_root, v_id)
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)
    print(dst_path)
    extract_frames(v_path, dst_path)
    print("finish video id: " + v_id)