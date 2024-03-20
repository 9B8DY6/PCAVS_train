data_path='./mvlrs_v1/main'

python -u train.py  \
        --name train_demo \
        --data_path ${data_path} \
        --dataset_mode lrs2train \
        --netG modulate \
        --netA resseaudio \
        --netA_sync ressesync \
        --netD multiscale \
        --netV resnext \
        --netE fan \
        --model av \
        --gpu_ids 0 \
        --clip_len 1 \
        --batchSize 2 \
        --style_dim 2560 \
        --nThreads 4 \
        --input_id_feature \
        --generate_interval 1 \
        --use_audio 1 \
        --noise_pose \
        --generate_from_audio_only \
        --no_id_loss \
        --epochs 3 \
        --gen_video \
