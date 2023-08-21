dataset=$1;

cd utils

python extract_audios.py --$dataset

python extract_frames.py --$dataset
