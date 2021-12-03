import argparse
import os
import logging
import traceback

import numpy as np
import matplotlib.pyplot as plt
from librosa import load, stft, griffinlim
from librosa.feature import mfcc, melspectrogram
from librosa.feature.inverse import mfcc_to_audio, mel_to_audio
from soundfile import write

from utils import *

logging.basicConfig(level=logging.INFO, format="%(asctime)s \t %(message)s", datefmt="%Y-%m-%d %H:%M:%S")


def hidden_voice_command_attack(feature, command_csv, dir, num_iterations, interval):
    feature_parameters = get_feature_parameters(feature)

    # The headers of the CSV file are defined as the absolute path and transcription of each and every command file.
    command_table = read_csv(csv_path=command_csv)
    hidden_voice_command_table = [["audio path", "transcription", "loss", "SNR"]]

    for command_info in command_table[1: ]:
        command_name = os.path.basename(command_info[0])[: -4]
        hidden_voice_command_dir = os.path.join(dir, "{}-{}".format(command_name, feature))

        if not os.path.exists(hidden_voice_command_dir):
            os.makedirs(hidden_voice_command_dir)
        
        command, _ = load(path=command_info[0], sr=16000)
        
        for feature_parameters_dict in feature_parameters:
            command_feature = audio_to_feature(command, feature, feature_parameters_dict)
            hidden_voice_command = command
            hidden_voice_command_feature = command_feature

            loss_trend = np.zeros(num_iterations)
            snr_trend = np.zeros(num_iterations)

            try:
                for i in range(num_iterations):
                    hidden_voice_command_path = os.path.join(hidden_voice_command_dir, 
                                                            "{}-{}-{}-{}-{}-{}.wav".format(command_name, 
                                                                                           feature_parameters_dict.get("n_mfcc", ""), 
                                                                                           feature_parameters_dict.get("n_mels", ""), 
                                                                                           feature_parameters_dict.get("n_fft"), 
                                                                                           feature_parameters_dict.get("hop_length"), i + 1))
                    
                    logging.info("Start to generate {}".format(hidden_voice_command_path))

                    hidden_voice_command = feature_to_audio(hidden_voice_command_feature, feature, feature_parameters_dict)
                    hidden_voice_command_feature = audio_to_feature(hidden_voice_command, feature, feature_parameters_dict)

                    loss = np.linalg.norm(hidden_voice_command_feature - command_feature)
                    snr = get_snr(audio=command, audio_with_noise=hidden_voice_command)

                    loss_trend[i] = loss
                    snr_trend[i] = snr

                    if (i + 1) % interval == 0:
                        hidden_voice_command_table.append([hidden_voice_command_path, command_info[1], format(loss, '.4f'), format(snr, '.2f')])
                        write(file=hidden_voice_command_path, data=hidden_voice_command, samplerate=16000)
                    
                    logging.info("loss = {:.4f}, snr = {:.2f}".format(loss, snr))
                
                fig, ax = plt.subplots(nrows=1, ncols=2, sharex='row')
                ax[0].plot(loss_trend)
                ax[1].plot(snr_trend)
                fig.savefig("{}.png".format(hidden_voice_command_path))
            except:
                logging.error("Error in {}".format(hidden_voice_command_path))
                traceback.print_exc()
            finally:
                logging.info("")
    
    write_csv(csv_path=os.path.join(dir, "hidden-voice-commands-{}.csv".format(feature)), lines=hidden_voice_command_table)


def audio_to_feature(audio, feature, feature_parameters_dict):
    if feature == 1:
        return np.abs(stft(y=audio, n_fft=feature_parameters_dict["n_fft"], hop_length=feature_parameters_dict.get("hop_length")))
    elif feature == 2:
        return melspectrogram(y=audio, sr=16000, 
                              n_mels=feature_parameters_dict["n_mels"], 
                              n_fft=feature_parameters_dict["n_fft"], 
                              hop_length=feature_parameters_dict.get("hop_length"))
    else:
        return mfcc(y=audio, sr=16000, 
                    n_mfcc=feature_parameters_dict["n_mfcc"], 
                    n_mels=feature_parameters_dict["n_mels"], 
                    n_fft=feature_parameters_dict["n_fft"], 
                    hop_length=feature_parameters_dict.get("hop_length"))


def feature_to_audio(audio_feature, feature, feature_parameters_dict):
    if feature == 1:
        return griffinlim(S=audio_feature, n_iter=16, hop_length=feature_parameters_dict.get("hop_length"))
    elif feature == 2:
        return mel_to_audio(M=audio_feature, sr=16000, 
                            n_fft=feature_parameters_dict["n_fft"], 
                            hop_length=feature_parameters_dict.get("hop_length"))
    else:
        return mfcc_to_audio(mfcc=audio_feature, sr=16000, 
                             n_mels=feature_parameters_dict["n_mels"], 
                             n_fft=feature_parameters_dict["n_fft"], 
                             hop_length=feature_parameters_dict.get("hop_length"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-f", "--feature", type=int, default=1, help="")
    parser.add_argument("-c", "--command_csv", type=str, default="commands/commands.csv", help="")
    parser.add_argument("-d", "--dir", type=str, default="hidden-voice-commands/", help="")
    parser.add_argument("-n", "--num_iterations", type=int, default=10000, help="")
    parser.add_argument("-i", "--interval", type=int, default=1000, help="")

    args = parser.parse_args()

    hidden_voice_command_attack(feature=args.feature, command_csv=args.command_csv, dir=args.dir, 
                                num_iterations=args.num_iterations, interval=args.interval)
