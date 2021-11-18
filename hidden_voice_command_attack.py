import argparse
import os
import logging
import traceback

import numpy as np
from librosa import load, stft, griffinlim
from librosa.feature import mfcc, melspectrogram
from librosa.feature.inverse import mfcc_to_audio, mel_to_audio
from soundfile import write

from utils import read_csv, write_csv

logging.basicConfig(level=logging.INFO, format="%(asctime)s \t %(message)s", datefmt="%Y-%m-%d %H:%M:%S")


def hidden_voice_command_attack(command_csv, feature, num_iterations):
    assert feature in (1, 2, 3)

    feature_parameters = get_feature_parameters(feature)

    # The headers of the CSV file are defined as the absolute path and transcription of each and every command file.
    command_table = read_csv(csv_path=command_csv)
    hidden_voice_command_table = list()

    for command_info in command_table:
        command_dir = os.path.dirname(command_info[0])
        command_name = os.path.basename(command_info[0])[: -4]
        hidden_voice_command_dir = os.path.join(command_dir, command_name)

        if not os.path.exists(hidden_voice_command_dir):
            os.makedirs(hidden_voice_command_dir)
            
        command, sr = load(path=command_info[0], sr=16000)
        
        for feature_parameters_dict in feature_parameters:
            command_feature = audio_to_feature(command, sr, feature, feature_parameters_dict)
            hidden_voice_command = command
            hidden_voice_command_feature = command_feature

            try:
                for i in range(num_iterations):
                    hidden_voice_command_path = os.path.join(hidden_voice_command_dir, 
                                                            "{}-{}-{}-{}-{}.wav".format(command_name, 
                                                                                        feature_parameters_dict.get("n_mfcc", ""), 
                                                                                        feature_parameters_dict.get("n_mels", ""), 
                                                                                        feature_parameters_dict.get("n_fft", ""), i + 1))
                    
                    logging.info("Start to generate {}".format(hidden_voice_command_path))

                    hidden_voice_command = feature_to_audio(hidden_voice_command_feature, sr, feature, feature_parameters_dict)
                    hidden_voice_command_feature = audio_to_feature(hidden_voice_command, sr, feature, feature_parameters_dict)

                    logging.info("loss = {:.4f}".format(np.linalg.norm(hidden_voice_command_feature - command_feature)))

                    hidden_voice_command_table.append([hidden_voice_command_path, command_info[1]])
                    write(file=hidden_voice_command_path, data=hidden_voice_command, samplerate=16000)
            except:
                logging.error("Error in {}".format(hidden_voice_command_path))
                traceback.print_exc()
            finally:
                logging.info("")
    
    write_csv(csv_path=os.path.join(command_dir, "hidden-voice-commands-{}.csv".format(feature)), lines=hidden_voice_command_table)


def get_feature_parameters(feature):
    feature_parameters = list()

    n_mfcc_list = (12, 13, 20, 26, 40)
    n_mels_list = (64, 128, 256, 512)
    n_fft_list = (512, 1024, 2048)

    if feature == 1:
        for n_fft in n_fft_list:
            feature_parameters.append({"n_fft": n_fft})
    elif feature == 2:
        for n_mels in n_mels_list:
            for n_fft in n_fft_list:
                feature_parameters.append({"n_mels": n_mels, "n_fft": n_fft})
    elif feature == 3:
        for n_mfcc in n_mfcc_list:
            for n_mels in n_mels_list:
                for n_fft in n_fft_list:
                    feature_parameters.append({"n_mfcc": n_mfcc, "n_mels": n_mels, "n_fft": n_fft})
    
    return feature_parameters


def audio_to_feature(audio, sr, feature, feature_parameters_dict):
    if feature == 1:
        return np.abs(stft(y=audio, n_fft=feature_parameters_dict["n_fft"]))
    elif feature == 2:
        return melspectrogram(y=audio, sr=sr, 
                              n_mels=feature_parameters_dict["n_mels"], 
                              n_fft=feature_parameters_dict["n_fft"])
    elif feature == 3:
        return mfcc(y=audio, sr=sr, 
                    n_mfcc=feature_parameters_dict["n_mfcc"], 
                    n_mels=feature_parameters_dict["n_mels"], 
                    n_fft=feature_parameters_dict["n_fft"])


def feature_to_audio(audio_feature, sr, feature, feature_parameters_dict):
    if feature == 1:
        return griffinlim(S=audio_feature)
    elif feature == 2:
        return mel_to_audio(M=audio_feature, sr=sr, 
                            n_fft=feature_parameters_dict["n_fft"])
    elif feature == 3:
        return mfcc_to_audio(mfcc=audio_feature, sr=sr, 
                             n_mels=feature_parameters_dict["n_mels"], 
                             n_fft=feature_parameters_dict["n_fft"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-c", "--command_csv", type=str, default="hidden-voice-commands/commands.csv", help="")
    parser.add_argument("-f", "--feature", type=int, default=3, help="")
    parser.add_argument("-n", "--num_iterations", type=int, default=5, help="")

    args = parser.parse_args()

    hidden_voice_command_attack(command_csv=args.command_csv, feature=args.feature, num_iterations=args.num_iterations)
