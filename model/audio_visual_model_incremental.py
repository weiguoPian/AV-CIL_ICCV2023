import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import LSCLinear, SplitLSCLinear


class IncreAudioVisualNet(nn.Module):
    def __init__(self, args, step_out_class_num, LSC=False):
        super(IncreAudioVisualNet, self).__init__()
        self.args = args
        self.modality = args.modality
        self.num_classes = step_out_class_num
        if self.modality != 'visual' and self.modality != 'audio' and self.modality != 'audio-visual':
            raise ValueError('modality must be \'visual\', \'audio\' or \'audio-visual\'')
        if self.modality == 'visual':
            self.visual_proj = nn.Linear(768, 768)
        elif self.modality == 'audio':
            self.audio_proj = nn.Linear(768, 768)
        else:
            self.audio_proj = nn.Linear(768, 768)
            self.visual_proj = nn.Linear(768, 768)
            self.attn_audio_proj = nn.Linear(768, 768)
            self.attn_visual_proj = nn.Linear(768, 768)
        
        if LSC:
            self.classifier = LSCLinear(768, self.num_classes)
        else:
            self.classifier = nn.Linear(768, self.num_classes)
    
    def forward(self, visual=None, audio=None, out_logits=True, out_features=False, out_features_norm=False, out_feature_before_fusion=False, out_attn_score=False, AFC_train_out=False):
        if self.modality == 'visual':
            if visual is None:
                raise ValueError('input frames are None when modality contains visual')
            visual_feature = torch.mean(visual, dim=1)
            visual_feature = F.relu(self.visual_proj(visual_feature))
            logits = self.classifier(visual_feature)
            outputs = ()
            if AFC_train_out:
                visual_feature.retain_grad()
                outputs += (logits, visual_feature)
                return outputs
            else:
                if out_logits:
                    outputs += (logits,)
                if out_features:
                    outputs += (F.normalize(visual_feature),)
                if len(outputs) == 1:
                    return outputs[0]
                else:
                    return outputs

        elif self.modality == 'audio':
            if audio is None:
                raise ValueError('input audio are None when modality contains audio')
            audio_feature = F.relu(self.audio_proj(audio))
            logits = self.classifier(audio_feature)
            outputs = ()
            if AFC_train_out:
                audio_feature.retain_grad()
                outputs += (logits, audio_feature)
                return outputs
            else:
                if out_logits:
                    outputs += (logits,)
                if out_features:
                    outputs += (F.normalize(audio_feature),)
                if len(outputs) == 1:
                    return outputs[0]
                else:
                    return outputs
        else:
            if visual is None:
                raise ValueError('input frames are None when modality contains visual')
            if audio is None:
                raise ValueError('input audio are None when modality contains audio')

            visual = visual.view(visual.shape[0], 8, -1, 768)
            spatial_attn_score, temporal_attn_score = self.audio_visual_attention(audio, visual)
            visual_pooled_feature = torch.sum(spatial_attn_score * visual, dim=2)
            visual_pooled_feature = torch.sum(temporal_attn_score * visual_pooled_feature, dim=1)
            
            audio_feature = F.relu(self.audio_proj(audio))
            visual_feature = F.relu(self.visual_proj(visual_pooled_feature))
            audio_visual_features = visual_feature + audio_feature
            
            logits = self.classifier(audio_visual_features)
            outputs = ()
            if AFC_train_out:
                audio_feature.retain_grad()
                visual_feature.retain_grad()
                visual_pooled_feature.retain_grad()
                outputs += (logits, visual_pooled_feature, audio_feature, visual_feature)
                return outputs
            else:
                if out_logits:
                    outputs += (logits,)
                if out_features:
                    if out_features_norm:
                        outputs += (F.normalize(audio_visual_features),)
                    else:
                        outputs += (audio_visual_features,)
                if out_feature_before_fusion:
                    outputs += (F.normalize(audio_feature), F.normalize(visual_feature))
                if out_attn_score:
                    outputs += (spatial_attn_score, temporal_attn_score)
                if len(outputs) == 1:
                    return outputs[0]
                else:
                    return outputs

    def audio_visual_attention(self, audio_features, visual_features):

        proj_audio_features = torch.tanh(self.attn_audio_proj(audio_features))
        proj_visual_features = torch.tanh(self.attn_visual_proj(visual_features))

        # (BS, 8, 14*14, 768)
        spatial_score = torch.einsum("ijkd,id->ijkd", [proj_visual_features, proj_audio_features])
        # (BS, 8, 14*14, 768)
        spatial_attn_score = F.softmax(spatial_score, dim=2)
        # (BS, 8, 768)
        spatial_attned_proj_visual_features = torch.sum(spatial_attn_score * proj_visual_features, dim=2)

        # (BS, 8, 768)
        temporal_score = torch.einsum("ijd,id->ijd", [spatial_attned_proj_visual_features, proj_audio_features])
        temporal_attn_score = F.softmax(temporal_score, dim=1)

        return spatial_attn_score, temporal_attn_score
    

    def incremental_classifier(self, numclass):
        weight = self.classifier.weight.data
        bias = self.classifier.bias.data
        in_features = self.classifier.in_features
        out_features = self.classifier.out_features

        self.classifier = nn.Linear(in_features, numclass, bias=True)
        self.classifier.weight.data[:out_features] = weight
        self.classifier.bias.data[:out_features] = bias
