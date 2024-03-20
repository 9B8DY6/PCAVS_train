# PCAVS_train
[Pose-Controllable Talking Face Generation by Implicitly Modularized Audio-Visual Representation (CVPR 2021)](https://github.com/Hangz-nju-cuhk/Talking-Face_PC-AVS)
![image](https://github.com/9B8DY6/PCAVS_train/assets/67573223/046083ca-78c0-42f2-9af2-7aedee38d612)

I wrote train code for pc-avs train.

## What is modified?
- PC-AVS generator → StyleGAN2-ada official pytorch generator
- Dataloader for LSR2 dataset
- train & result save code

## Docker build
```
cd PCAVS_train
docker build -t talkface
docker run -it --gpus all --name talkface1 -v {PCAVS_train_folder_location}:/workspace --shm-size 128g talkface:latest /bin/bash
```
## Train
```
bash experimetns/train.sh
```

