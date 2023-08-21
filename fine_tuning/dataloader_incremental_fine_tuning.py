import numpy as np
import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import h5py

class IcreLoader(Dataset):
    def __init__(self, args, mode='train', modality='visual', incremental_step=0):
        self.mode = mode
        self.args = args
        self.modality = modality

        if args.dataset == 'AVE':
            self.data_root = '../data/AVE'
            self.visual_pretrained_feature_path = os.path.join(self.data_root, 'visual_pretrained_feature', 'visual_pretrained_feature_dict.npy')
            self.all_visual_pretrained_features = np.load(self.visual_pretrained_feature_path, allow_pickle=True).item()
        else:
            if args.dataset == 'ksounds':
                self.data_root = '../data/kinetics-sounds'
            elif args.dataset == 'VGGSound_100':
                self.data_root = '../data/VGGSound_100'
            self.visual_pretrained_feature_path = os.path.join(self.data_root, 'visual_pretrained_feature', 'visual_features.h5')
            self.all_visual_pretrained_features = h5py.File(self.visual_pretrained_feature_path, 'r')

        self.audio_pretrained_feature_path = os.path.join(self.data_root, 'audio_pretrained_feature', 'audio_pretrained_feature_dict.npy')
        self.all_audio_pretrained_features = np.load(self.audio_pretrained_feature_path, allow_pickle=True).item()

        self.all_id_category_dict = np.load(
            os.path.join(self.data_root, 'all_id_category_dict.npy'), allow_pickle=True
        ).item()
        self.categoty_encode_dict = np.load(
            os.path.join(self.data_root, 'category_encode_dict.npy'), allow_pickle=True
        ).item()
        self.all_classId_vid_dict = np.load(
            os.path.join(self.data_root, 'all_classId_vid_dict.npy'), allow_pickle=True
        ).item()

        if self.mode == 'train':
            self.all_id_category_dict = self.all_id_category_dict['train']
            self.all_classId_vid_dict = self.all_classId_vid_dict['train']
        elif self.mode == 'val':
            self.all_id_category_dict = self.all_id_category_dict['val']
            self.all_classId_vid_dict = self.all_classId_vid_dict['val']
        elif self.mode == 'test':
            self.all_id_category_dict = self.all_id_category_dict['test']
            self.all_classId_vid_dict = self.all_classId_vid_dict['test']
        else:
            raise ValueError('mode must be \'train\', \'val\' or \'test\'')
        
        if self.modality != 'visual' and self.modality != 'audio' and self.modality != 'audio-visual':
            raise ValueError('modality must be \'visual\', \'audio\' or \'audio-visual\'')

        self.incremental_step = incremental_step
        self.current_step_class = self.set_current_step_classes()
        self.all_current_data_vids = self.current_step_data()

    def set_current_step_classes(self):
        if self.mode == 'train':
            if self.args.upper_bound:
                current_step_class = np.array(range(0, self.args.class_num_per_step * (self.incremental_step + 1)))
            else:
                current_step_class = np.array(range(self.args.class_num_per_step * self.incremental_step, self.args.class_num_per_step * (self.incremental_step + 1)))
        else:
            current_step_class = np.array(range(0, self.args.class_num_per_step * (self.incremental_step + 1)))
        return current_step_class

    def current_step_data(self):
        all_current_data_vids = []
        for class_idx in self.current_step_class:
            # print(class_idx)
            # print(list(self.all_classId_vid_dict.keys()))
            all_current_data_vids += self.all_classId_vid_dict[class_idx]
        return all_current_data_vids

    def set_incremental_step(self, step):
        self.incremental_step = step
        self.current_step_class = self.set_current_step_classes()
        self.all_current_data_vids = self.current_step_data()

    def __getitem__(self, index):
        vid = self.all_current_data_vids[index]
        category = self.all_id_category_dict[vid]
        category_id = self.categoty_encode_dict[category]
        
        if 'visual' in self.modality:
            if self.args.dataset == 'AVE':
                visual_feature = self.all_visual_pretrained_features[vid]
            else:
                visual_feature = self.all_visual_pretrained_features[vid][()]
            visual_feature = torch.Tensor(visual_feature)
        
        if 'audio' in self.modality:
            audio_feature = self.all_audio_pretrained_features[vid]
            audio_feature = torch.Tensor(audio_feature)
        
        if self.modality == 'visual':
            return visual_feature, category_id
        elif self.modality == 'audio':
            return audio_feature, category_id
        else:
            return (visual_feature, audio_feature), category_id

    def close_visual_features_h5(self):
        self.all_visual_pretrained_features.close()

    def __len__(self):
        return len(self.all_current_data_vids)