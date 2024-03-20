
import os
import sys
#import math
import numpy as np
from config import AudioConfig
#import shutil
import cv2
#import glob
import ffmpeg
import librosa
#import random
from random import sample
import torch
from data.base_dataset import BaseDataset

import util.util as util
from scripts.align_68 import align_batches
#from IPython.core.debugger import set_trace

class LRS2trainDataset(BaseDataset):
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--no_pairing_check', action='store_true',
                            help='If specified, skip sanity check of correct label-image file pairing')
        return parser

    def cv2_loader(self, img_str):
        img_array = np.frombuffer(img_str, dtype=np.uint8)
        return cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    def load_img(self, img, M=None, crop=True, crop_len=16):

        if img is None:
            raise Exception('None Image')

        if M is not None:
            img = cv2.warpAffine(img, M, (self.opt.crop_size, self.opt.crop_size), borderMode=cv2.BORDER_REPLICATE)

        if crop:
            img = img[:self.opt.crop_size - crop_len*2, crop_len:self.opt.crop_size - crop_len]
            if self.opt.target_crop_len > 0:
                img = img[self.opt.target_crop_len:self.opt.crop_size - self.opt.target_crop_len, self.opt.target_crop_len:self.opt.crop_size - self.opt.target_crop_len]
            img = cv2.resize(img, (self.opt.crop_size, self.opt.crop_size))

        return img


    def frame2audio_indexs(self, frame_inds):
        start_frame_ind = frame_inds - self.audio.num_frames_per_clip // 2

        start_audio_inds = start_frame_ind * self.audio.num_bins_per_frame
        return start_audio_inds

    def audio_normalize(self,samples, desired_rms=0.1, eps=1e-4):
        rms = np.maximum(eps, np.sqrt(np.mean(samples ** 2)))
        samples = samples * (desired_rms / rms)
        return samples

    def decode_audio(self, in_filename, **input_kwargs):
        try:
            input_video = ffmpeg.input(in_filename)
            input_video.audio
            out, err = ( ffmpeg
                .input(in_filename)
                .output('-', format='s16le', acodec='pcm_s16le', ac=1, ar='16k')
                .global_args('-y', '-loglevel',"error")
                .overwrite_output()
                .run(capture_stdout=True)
                )
            pcm_data = np.frombuffer(out, dtype = 'int16')
            outt = self.audio_normalize(librosa.util.buf_to_float(pcm_data, 2))
        except ffmpeg.Error as e:
            print(e.stderr, file=sys.stderr)
            sys.exit(1)
        return outt


    def mp4_to_batch_npy(self, in_filename):
        probe = ffmpeg.probe(in_filename)
        video_info = next(x for x in probe['streams'] if x['codec_type'] == 'video')
        width = int(video_info['width'])
        height = int(video_info['height'])
        num_frames = int(video_info['nb_frames'])

        out, _ = (    ffmpeg
            .input(in_filename, ss=0.0)
            .output('-',format='rawvideo', pix_fmt='rgb24', **{'qscale:v': 2})
            .global_args('-y', '-loglevel', "error")
            .run(capture_stdout=True)
        )
        video = (
            np
            .frombuffer(out, np.uint8)
            .reshape([-1, height, width, 3])
        )
        return video

    def initialize(self, opt):
        self.opt = opt
        self.path_label = opt.path_label
        self.clip_len = opt.clip_len
        self.frame_interval = opt.frame_interval
        self.num_clips = opt.num_clips
        self.frame_rate = opt.frame_rate
        self.num_inputs = opt.num_inputs
        self.filename_tmpl = opt.filename_tmpl ## {.6%d}.jpg

        self.mouth_num_frames = None
        self.mouth_frame_path = None
        self.pose_num_frames = None

        self.audio = AudioConfig.AudioConfig(num_frames_per_clip=opt.num_frames_per_clip, hop_size=opt.hop_size)
        self.num_audio_bins = self.audio.num_frames_per_clip * self.audio.num_bins_per_frame

        ### opt.path_label  ./mrlvs_v1/main/숫자나열/00001.mp4

        if not opt.isTrain: 
            id_idx = str(opt.path_label.split('/')[-2])
            video_in_id_idx = str(opt.path_label.split('/')[-1].split('.')[0])
            self.processed_file_savepath = os.path.join('results', 'id_' + id_idx + '_video_num_' + video_in_id_idx)
            if not os.path.exists(self.processed_file_savepath): os.makedirs(self.processed_file_savepath)


        wav = self.decode_audio(self.path_label)
        self.spectrogram = self.audio.audio_to_spectrogram(wav)

        self.target_frame_inds = np.arange(2, len(self.spectrogram) // self.audio.num_bins_per_frame - 2)
        self.audio_inds = self.frame2audio_indexs(self.target_frame_inds)

        self.dataset_size = len(self.target_frame_inds)

        video_into_batch_npy = self.mp4_to_batch_npy(self.path_label)
        self.num_target_frame = video_into_batch_npy.shape[0]
        self.face_aligned = align_batches(video_into_batch_npy) ## (frame, 224, 224, 3)
        self.augmented = self.face_augmentation(self.face_aligned, opt.crop_size) ##(frame,224,224,3) but augmented

        num_frame = self.face_aligned.shape[0]
        rand_idx = sample(range(num_frame),opt.num_inputs) ##list
        id_img_tensors = []
        for idx in rand_idx:
            id_img_tensor = self.to_Tensor(self.load_img(self.face_aligned[idx]))
            id_img_tensors += [id_img_tensor]


        self.id_img_tensor = torch.stack(id_img_tensors)
        self.initialized = False

    def load_spectrogram(self, audio_ind):
        mel_shape = self.spectrogram.shape

        if (audio_ind + self.num_audio_bins) <= mel_shape[0] and audio_ind >= 0:
            spectrogram = np.array(self.spectrogram[audio_ind:audio_ind + self.num_audio_bins, :]).astype('float32')
        else:
            print('(audio_ind {} + opt.num_audio_bins {}) > mel_shape[0] {} '.format(audio_ind, self.num_audio_bins,
                                                                                     mel_shape[0]))
            if audio_ind > 0:
                spectrogram = np.array(self.spectrogram[audio_ind:audio_ind + self.num_audio_bins, :]).astype('float32')
            else:
                spectrogram = np.zeros((self.num_audio_bins, mel_shape[1])).astype(np.float16).astype(np.float32)

        spectrogram = torch.from_numpy(spectrogram)
        spectrogram = spectrogram.unsqueeze(0)

        spectrogram = spectrogram.transpose(-2, -1)
        return spectrogram

    def __getitem__(self, index):

        img_index = self.target_frame_inds[index]
        mel_index = self.audio_inds[index]

        target_index = util.calc_loop_idx(img_index,self.num_target_frame)

        target_frame = self.to_Tensor(self.load_img(self.face_aligned[target_index]))
        augmented_frame = self.to_Tensor(self.load_img(self.augmented[target_index]))

        spectrograms = self.load_spectrogram(mel_index)

        input_dict = {
                      'input': self.id_img_tensor,
                      'target': target_frame,
                      'augmented': augmented_frame,
                      'label': torch.zeros(1),
                      }
        if self.opt.use_audio:
            input_dict['spectrograms'] = spectrograms

        # Give subclasses a chance to modify the final output
        self.postprocess(input_dict)

        return input_dict

    def postprocess(self, input_dict):
        return input_dict

    def __len__(self):
        return self.dataset_size

    def get_processed_file_savepath(self):
        return self.processed_file_savepath
