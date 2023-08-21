# Audio-Visual Class-Incremental Learning 

<!-- [Paper]() -->

We introduce <b>audio-visual class-incremental learning</b>, a class-incremental learning scenario for audio-visual video recognition, and propose a method <b>AV-CIL</b>.

<div align="center">
  <img width="100%" alt="AV-CIL" src="images/model.png">
</div>


## Environment

We conduct experiments with Python 3.8.13 and Pytorch 1.13.0.

To setup the environment, please simply run

```
pip install -r requirements.txt
```

## Datasets

### AVE

The original AVE dataset can be downloaded through [link](https://drive.google.com/open?id=1FjKwe79e0u96vdjIVwfRQ1V6SoDHe7kK).


### Kinetics-Sounds

The original Kinetics dataset can be downloaded through [link](https://github.com/cvdfoundation/kinetics-dataset). After downloading the Kinetics dataset, please apply our provided video id list ([here](./data/kinetics-sounds/)) to extract the Kinetics-Sounds dataset used in our experiments.


### VGGSound100

The original VGGSound dataset can be downloaded through [link](https://www.robots.ox.ac.uk/~vgg/data/vggsound/). After downloading the VGGSound dataset, please apply our provided video id list ([here](./data/VGGSound_100/)) to extract the Kinetics-Sounds dataset used in our experiments.

## Feature extraction

To be updated...


## Training & Evaluation

For vanilla fine-tuning strategy, please run

```
sh run_incremental_fine_tuning.sh 'dataset' 'modality'
```

The 'dataset' should be in [AVE, ksounds, VGGSound_100], and the 'modality' should be in [audio, visual, audio-visual].


For the upper bound, please run

```
sh run_incremental_upper_bound.sh 'dataset' 'modality'
```


For LwF, please run

```
sh run_incremental_lwf.sh 'dataset' 'modality'
```


For iCaRL, please run

```
sh run_incremental_lwf.sh 'dataset' 'modality' 'classifier'
```

The 'classifier' should be in [NME, FC].



For SS-IL, please run

```
sh run_incremental_ssil.sh 'dataset' 'modality'
```



For AFC, please run

```
sh run_incremental_afc.sh 'dataset' 'modality' 'classifier'
```

The 'classifier' should be in [NME, LSC].




For our AV-CIL, please run

```
sh run_incremental_ours.sh 'dataset'
```



## Citation

If you find this work useful, please consider citing it.
```
@inproceedings{pian2023audiovisual,
  title={Audio-Visual Class-Incremental Learning},
  author={Pian, Weiguo and Mo, Shentong and Guo, Yunhui and Tian, Yapeng},
  booktitle={IEEE/CVF International Conference on Computer Vision},
  year={2023}
}
```