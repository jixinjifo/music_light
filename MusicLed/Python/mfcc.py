import numpy as np
import matplotlib.pyplot as plt
import time
import math
from scipy.fftpack import dct
import config

def HzToMel(f):
    return 1127 * np.log(1.0 + f / 700)

def MelToHz(m):
    return 700 * (np.exp(m / 1127) - 1)

def calc_filters(nfilters, nfft, sample_rate, freq_min, freq_max):
    valid_nfft = valid_nfft = int(nfft / 2 + 1)
    filters = np.zeros((nfilters, int(np.floor(valid_nfft))))
    low_mel = HzToMel(freq_min)
    high_mel = HzToMel(freq_max)
    mel_bw = (high_mel - low_mel) / (nfilters + 1)
    fre_bin = sample_rate / nfft
    for j in range(1, nfilters + 1):
        mel_cent = j * mel_bw + low_mel
        mel_left  = mel_cent - mel_bw
        mel_right = mel_cent + mel_bw
        freq_cent =  MelToHz(mel_cent)
        freq_left =  MelToHz(mel_left)
        freq_bw_left = freq_cent - freq_left
        freq_right = MelToHz(mel_right)
        freq_bw_right = freq_right - freq_cent
        for i in range(1, valid_nfft + 1):
            freq = (i-1) * fre_bin
            if freq_right > freq > freq_left:
                if freq <= freq_cent:
                    filters[j-1][i-1] = (freq - freq_left) / freq_bw_left
                else:
                    filters[j-1][i-1] = (freq_right - freq) / freq_bw_right
    return filters

def calc_mfcc(data, sample_rate, frame_len, frame_shift, nfft, nfilters, preF, freq_min, freq_max):
    if preF:
        data = np.append(data[0], data[1:] - 0.97 * data[:-1]) # 预加重
    data_len = len(data) # 信号长度
    nframes = math.ceil((data_len - frame_len) / frame_shift) + 1 # 信号帧数
    pad_data_len = (nframes - 1) * frame_shift + frame_len # 补齐0后信号所占长度
    z = np.zeros((pad_data_len - data_len)) # 需要补齐的0
    pad_data = np.append(data, z) # 补齐0后的信号
    indices = np.tile(np.arange(0, frame_len), (nframes, 1)) + \
    np.tile(np.arange(0, nframes * frame_shift, frame_shift), (frame_len, 1)).T
    frame_data = pad_data[indices.astype(np.int32, copy=False)] # 将信号转换成(帧数, 每帧所占长度)的数组
    window = (0.54 - 0.46 * np.cos((2 * np.pi * np.arange(1, frame_len + 1)) / (frame_len + 1)))
    frame_data = frame_data * window # 加窗后的信号
    mag_frames = np.abs(np.fft.rfft(frame_data, nfft))
    pow_frames = mag_frames ** 2 # 确认是否归一化 / nfft?
    filters = calc_filters(nfilters, nfft, sample_rate, freq_min, freq_max)
    filter_banks = np.dot(pow_frames, filters.T)
    # filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    # filter_banks = np.log10(filter_banks)  # 转换为分贝
    # mfcc = dct(filter_banks, type=2, axis=-1, norm=None)
    mfcc = filter_banks
    return mfcc

if __name__ == '__main__':
	filters = calc_filters(config.N_FFT_BINS, config.MIC_RATE/config.FPS, config.MIC_RATE, config.MIN_FREQUENCY, config.MAX_FREQUENCY)
	print(np.sort(filters))