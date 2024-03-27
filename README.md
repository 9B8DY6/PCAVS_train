# PCAVS_train
[Pose-Controllable Talking Face Generation by Implicitly Modularized Audio-Visual Representation (CVPR 2021)](https://github.com/Hangz-nju-cuhk/Talking-Face_PC-AVS)
![image](https://github.com/9B8DY6/PCAVS_train/assets/67573223/046083ca-78c0-42f2-9af2-7aedee38d612)
> We propose Pose-Controllable Audio-Visual System (PC-AVS), which achieves free pose control when driving arbitrary talking faces with audios. Instead of learning pose motions from audios, we leverage another pose source video to compensate only for head motions. The key is to devise an implicit low-dimension pose code that is free of mouth shape or identity information. In this way, audio-visual representations are modularized into spaces of three key factors: speech content, head pose, and identity information.

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

