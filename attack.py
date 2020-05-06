import numpy as np
import matplotlib.pyplot as plt
from librosa import load, get_duration, resample, stft
from librosa.feature import mfcc
from librosa.display import waveplot
from librosa.output import write_wav


def main():
    song_path = ROOT_DIRECTORY + "Song-44.1kHz.wav"
    command_path = ROOT_DIRECTORY + "Command.wav"
    attack(song_path, command_path)


def attack(song_path, command_path):
    song, song_sr = load(song_path, sr=16000)
    command, command_sr = load(command_path, sr=16000)
    #
    # duration = get_duration(y=command, sr=command_sr)
    # origin = 1
    # song = song[int(origin * song_sr): int((origin + duration) * song_sr)]
    #
    # song_mfcc = mfcc(y=resample(y=song, orig_sr=song_sr, target_sr=command_sr), sr=command_sr)
    # command_mfcc = mfcc(y=command, sr=command_sr)
    # epsilon = np.linalg.norm(0.01 * command_mfcc)
    # print(epsilon)
    # print(song_mfcc.shape)
    # print(command_mfcc.shape)

    # plot_mfcc(song_mfcc, command_mfcc)
    # write_wav(path=ROOT_DIRECTORY+"Song.wav", y=song, sr=song_sr)


if __name__ == '__main__':
    ROOT_DIRECTORY = "./Audio Samples/"
    main()
