# PCAVS_train
I wrote train code by myself for pc-avs training

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

