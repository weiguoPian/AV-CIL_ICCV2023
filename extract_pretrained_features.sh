dataset=$1;

cd utils

python gen_audio_features.py --$dataset

python gen_visual_features.py --$dataset



