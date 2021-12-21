import csv

import numpy as np


def read_csv(csv_path):
    with open(csv_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        lines = [line for line in csv_reader]

    return lines


def write_csv(csv_path, lines):
    with open(csv_path, 'w') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerows(lines)


def get_feature_parameters(feature):
    assert feature in (0, 1, 2, 3)
    
    feature_parameters = list()

    n_mfcc_list = (20, )
    n_mels_list = (40, 80, 128)
    n_fft_list = (400, 512)
    hop_length_ratio = (0.5, )

    if feature <= 1:
        for n_fft in n_fft_list:
            for hop_length in hop_length_ratio:
                feature_parameters.append({"n_fft": n_fft, "hop_length": int(n_fft * hop_length)})
    elif feature == 2:
        for n_mels in n_mels_list:
            for n_fft in n_fft_list:
                for hop_length in hop_length_ratio:
                    feature_parameters.append({"n_mels": n_mels, "n_fft": n_fft, "hop_length": int(n_fft * hop_length)})
    else:
        for n_mfcc in n_mfcc_list:
            for n_mels in n_mels_list:
                for n_fft in n_fft_list:
                    for hop_length in hop_length_ratio:
                        feature_parameters.append({"n_mfcc": n_mfcc, "n_mels": n_mels, "n_fft": n_fft, "hop_length": int(n_fft * hop_length)})
    
    return feature_parameters


def get_snr(audio, audio_with_noise):
    length = min(len(audio), len(audio_with_noise))
    
    audio = audio[: length]
    audio_with_noise = audio_with_noise[: length]
    noise = audio_with_noise - audio

    p_audio = np.sum(audio ** 2)
    p_noise = np.sum(noise ** 2)

    snr = 10 * np.log10(p_audio / p_noise)

    return snr
