import os
import sys
sys.path.append('..')
import glob
from options.train_options import TrainOptions

import torch
from models import create_model
import data
import util.util as util
from tqdm import tqdm
import ffmpeg

def video_concat(processed_file_savepath, name, video_names, audio_path):
    cmd = ['ffmpeg']
    num_inputs = len(video_names)
    for video_name in video_names:
        cmd += ['-i', '\'' + str(os.path.join(processed_file_savepath, video_name + '.mp4'))+'\'',]

    cmd += ['\'' + str(os.path.join(processed_file_savepath, name+'.mp4')) + '\'', '-loglevel error -y']
    cmd = ' '.join(cmd)
    os.system(cmd)

    video_add_audio(name, audio_path, processed_file_savepath)


def video_add_audio(name, audio_path, processed_file_savepath):
    video = ffmpeg.input(os.path.join(processed_file_savepath, name + '.mp4'))
    audio = ffmpeg.input(audio_path).audio

    out, err=(
        ffmpeg
        .concat(video, audio, v=1, a=1)
        .output(os.path.join(processed_file_savepath, 'av' + name + '.mp4'), strict='-2',**{'qscale:v': 0})
        .global_args('-y', '-loglevel',"error")
        .overwrite_output()
        .run(capture_stdout=True, capture_stderr=True)
    )


def img2video(dst_path, prefix, video_path):
    cmd = ['ffmpeg', '-i', '\'' + video_path + '/' + prefix + '%d.jpg'
           + '\'', '-q:v 0', '\'' + dst_path + '/' + prefix + '.mp4' + '\'', '-loglevel error -y']
    cmd = ' '.join(cmd)
    os.system(cmd)



def inference_single_audio(opt, path_label, model):
    opt.isTrain = False
    torch.manual_seed(0)
    with torch.no_grad():
        model.eval()
    
    opt.path_label = path_label
    dataloader = data.create_dataloader(opt)
    processed_file_savepath = dataloader.dataset.get_processed_file_savepath()
    
    idx = 0

    save_path = processed_file_savepath
    util.mkdir(save_path)

    for data_i in tqdm(dataloader):
        # print('==============', i, '===============')

        fake_image_original_pose_a, fake_image_driven_pose_a = model.forward(data_i, mode='inference')

        for num in range(len(fake_image_original_pose_a)):
            if opt.driving_pose:
                video_name = 'DrivenPose'
                util.save_torch_img(fake_image_driven_pose_a[num],
                         os.path.join(save_path, video_name + str(idx) + '.jpg'))
            else:
                video_name = 'OriginPose'
                util.save_torch_img(fake_image_original_pose_a[num],
                                    os.path.join(save_path, video_name + str(idx) + '.jpg'))
            idx += 1

    if opt.gen_video:
        img2video(processed_file_savepath, video_name, save_path)
        video_names = [video_name]
        audio_path = path_label
        video_concat(processed_file_savepath, 'video_audio_concat', video_names, audio_path)

    print('results saved...' + processed_file_savepath)
    del dataloader
    return

def train_single_audio(opt, path_label, model, op_G, op_D):
    opt.path_label = path_label
    dataloader = data.create_dataloader(opt)
    total_loss = 0

    for epoch in range(opt.epochs):
        for data_i in tqdm(dataloader):
            model.zero_grad()
            loss, images_list, _ = model.forward(data_i, mode='generator')
            total_loss = total_loss + torch.tensor(
                loss['GAN_Feat_audio'].clone().detach().requires_grad_(True)+
                loss['GANa'].clone().detach().requires_grad_(True)+
                loss['VGGa'].clone().detach().requires_grad_(True)+
                loss['CrossModal'].clone().detach().requires_grad_(True)).requires_grad_(True)
            total_loss.backward()
            op_D.step()
            op_G.step()

        if epoch % opt.display_freq ==0:
            print(total_loss/float(epoch+len(dataloader)))
            inference_single_audio(opt, path_label, model)
            opt.isTrain = True

    del dataloader
    return


def main():

    opt = TrainOptions().parse()
    opt.isTrain = True
    torch.manual_seed(0)
    model = create_model(opt).cuda()
    model.train()
    op_G, op_D = model.create_optimizers(opt)

    path_labels = glob.glob(f'{opt.data_path}/*/*.mp4')

    for clip_idx, path_label in enumerate(path_labels):

        train_single_audio(opt, path_label, model, op_G, op_D)

if __name__ == '__main__':
    main()
