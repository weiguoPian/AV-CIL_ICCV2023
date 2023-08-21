import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
from model.audioMAE import get_model

import torch
import torchaudio

import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='AVE', choices=['AVE', 'ksounds', 'VGGSound_100'])

args = parser.parse_args()


model = get_model()

pretrained_dict = torch.load('../model/pretrained/audioMAE_pretrained.pth', map_location='cpu')['model']
model.load_state_dict(pretrained_dict, strict=False)

model.eval()

audio_root = '../raw_data/{}/audios/'.format(args.dataset)
audio_name_list = os.listdir(audio_root)

audio_pretrained_feature_dict = {}
i = 0
for name in audio_name_list:
    print(i, name)
    audio_path = os.path.join(audio_root, name)
    waveform, sr = torchaudio.load(audio_path)
    waveform = waveform - waveform.mean()

    fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
                                              window_type='hanning', num_mel_bins=128, dither=0.0, frame_shift=10)
    p = 1024 - len(fbank)
    if p > 0:
        m = torch.nn.ZeroPad2d((0, 0, 0, p))
        fbank = m(fbank)
    elif p < 0:
        fbank = fbank[:1024, :]
    
    fbank = fbank.unsqueeze(dim=0).unsqueeze(dim=0)
    with torch.no_grad():
        out = model(fbank).squeeze(dim=0).detach().numpy()
    v_id = name.split('.')[0]
    audio_pretrained_feature_dict[v_id] = out
    i += 1

np.save('../data/{}/audio_pretrained_feature/audio_pretrained_feature_dict.npy'.format(args.dataset), audio_pretrained_feature_dict)

