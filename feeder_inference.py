import ipdb as pdb
import tensorflow as tf
import numpy as np
import time
from _thread import start_new_thread
import queue
from python_speech_features import logfbank
import vad_ex
import webrtcvad
import utils
import re
import os
import random
import pickle
import glob
import sys
import argparse
import io
import soundfile as sf


# random.randrange
"""
input dir

vox1_dev_wav - id #### - 0DOmwbPlPvY - 00001.wav
                                     - 00002.wav
                                     - ...
                       - 5VNK93duiOM
                       - ...
                       
             - id #### - ...

"""


class Feeder():
    def __init__(self, in_wav1, in_wav2):
        # Set hparams
        self.segment_length = 1.6
        self.overlap_ratio = 0.5
        self.in_wav1 = in_wav1
        self.in_wav2 = in_wav2
        # print("Wav_B1 = ", self.in_wav1[:1000])

        #self.placeholders = placeholders
    def preprocess(self):
        # self.hparams.in_wav1, self.hparams.in_wav2 are full paths of the wav file
        # for ex) /home/hdd2tb/ninas96211/dev_wav_set/id10343_pCDWKHjQjso_00002.wav

        wavs_list = [self.in_wav1, self.in_wav2]

        self.save_dict = {}

        ####
        # import wave
        # import io
        # import soundfile as sf
        #
        # with open(wav1, 'rb') as f:
        #     wav_b = f.read()                  ### wav_b is input from webserver
        # file_on_memory = io.BytesIO(wav_b)
        # data, sr = sf.read(file_on_memory)

        # file_name for ex) id10343_pCDWKHjQjso_00002
        for wav_id, wav_path in enumerate(wavs_list):
            file_on_memory = io.BytesIO(wav_path)
            audio, sample_rate = sf.read(file_on_memory)
            print("audio.shape:")
            print(audio.shape)
            wav_arr = np.asarray(audio, dtype=np.float32)[
                :, 0]  # streo to mono
            print("wav_arr.shape:")
            print(wav_arr.shape)
            # print("wav_arr")
            # print(wav_arr[:1000])

            # wav_id = wav_path.rstrip(".wav")1
            # audio, sample_rate = vad_ex.read_wave(file_on_memory)
            # print("wav_id : ", wav_id, "audio: ", audio[:10], "sample_rate: ", sample_rate )
            # vad = webrtcvad.Vad(1)
            # frames = vad_ex.frame_generator(30, audio, sample_rate)
            # frames = list(frames)
            # segments = vad_ex.vad_collector(sample_rate, 30, 300, vad, frames)
            # total_wav = b""
            # for i, segment in enumerate(segments):
            #     total_wav += segment
            #     print(wav_id+ " : " + str(i)+"th segment appended")
            # # Without writing, unpack total_wav into numpy [N,1] array
            # # 16bit PCM 기준 dtype=np.int16
            # wav_arr = np.frombuffer(total_wav, dtype=np.int16)
            print("read audio data from byte string. np array of shape:" +
                  str(wav_arr.shape))
            logmel_feats = logfbank(wav_arr, samplerate=sample_rate, nfilt=40)

            # print ("logmelfeats")
            # print(logmel_feats)
            # file_name for ex, 'id10343_pCDWKHjQjso_00002'
            self.save_dict[wav_id] = logmel_feats

    def create_data(self):

        # Create input batch of shape [num_dvectors, total_frames, 40(spectrogram_channel)]
        # N frames  - (N-1) overlaps  < total_len
        # max_N = (total_len - overlaps) // (frames - overlaps)

        num_frames = self.segment_length * 100
        num_overlap_frames = num_frames * self.overlap_ratio
        dvector_dict = {}

        match = False
        prev_wav_name = ""

        for wav_name, feats in self.save_dict.items():
            # if wav_name.split("_")[0] == prev_wav_name:
            #     match = True
            total_len = feats.shape[0]
            num_dvectors = int((total_len - num_overlap_frames) //
                               (num_frames - num_overlap_frames))
            print("num dvec:" + str(num_dvectors))
            dvectors = []
            for dvec_idx in range(num_dvectors):
                start_idx = int((num_frames - num_overlap_frames) * dvec_idx)
                end_idx = int(start_idx + num_frames)
                print("wavname: " + str(wav_name) +
                      " start_idx: " + str(start_idx))
                print("wavname: " + str(wav_name) +
                      " end_idx: " + str(end_idx))
                dvectors.append(feats[start_idx:end_idx, :])
            dvectors = np.asarray(dvectors, dtype=np.float32)
            dvector_dict[wav_name] = dvectors
            # prev_wav_name = wav_name.split("_")[0]

        wav1_data = list(dvector_dict.values())[0]
        wav2_data = list(dvector_dict.values())[1]

        print("match: " + str(match))
        print("wav1_data.shape:" + str(wav1_data.shape))
        print(wav1_data)
        print("wav2_data.shape:" + str(wav2_data.shape))
        return wav1_data, wav2_data, match


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--in_wav1", type=str,
                        required=True, help="input wav1 dir")
    parser.add_argument("--in_wav2", type=str,
                        required=True, help="input wav2 dir")
    #/home/hdd2tb/ninas96211/dev_wav_set

    #parser.add_argument("--ckpt_dir", type=str, required=True, help="checkpoint to start with for inference")

    # Data
    #parser.add_argument("--window_length", type=int, default=160, help="sliding window length(frames)")
    parser.add_argument("--segment_length", type=float,
                        default=0.8, help="segment length in seconds")
    parser.add_argument("--overlap_ratio", type=float,
                        default=0.5, help="overlaping percentage")
    parser.add_argument("--spectrogram_scale", type=int, default=40,
                        help="scale of the input spectrogram")
    # Enrol
    parser.add_argument("--num_spk_per_batch", type=int, default=5,
                        help="N speakers of batch size N*M")
    parser.add_argument("--num_utt_per_batch", type=int, default=10,
                        help="M utterances of batch size N*M")

    args = parser.parse_args()

    feeder = Feeder(args)

    feeder.preprocess()
    feeder.create_data()
