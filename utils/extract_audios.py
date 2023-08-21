import os
import moviepy
from moviepy.audio.AudioClip import AudioArrayClip
from moviepy.editor import VideoFileClip
import numpy as np
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='AVE', choices=['AVE', 'ksounds', 'VGGSound_100'])

args = parser.parse_args()


video_root = '../raw_data/{}/videos/'.format(args.dataset)

audio_root = '../raw_data/{}/audios/'.format(args.dataset)

if not os.path.exists(audio_root):
    os.makedirs(audio_root)

v_name_list = os.listdir(video_root)
for v_name in v_name_list:
    v_id = v_name[:-4]
    v_path = os.path.join(video_root, v_name)
    audio_name = v_id + '.wav'
    dst_path = os.path.join(audio_root, audio_name)
    try:
        video = VideoFileClip(v_path)
        audio = video.audio
        audio.write_audiofile(dst_path, fps=11430)
        print("finish video id: " + audio_name)
    except:
        print('Failed video id: {}' + audio_name)

