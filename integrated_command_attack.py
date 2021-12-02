import argparse
import os
import logging

import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
from torch.optim import Adam, SGD
from torchaudio import load, save
from torchaudio.transforms import MFCC, MelSpectrogram, Spectrogram

from utils import *

logging.basicConfig(level=logging.INFO, format="%(asctime)s \t %(message)s", datefmt="%Y-%m-%d %H:%M:%S")


def integrated_command_attack(feature, command_csv, song_dir, dir, interval, optimizer, penalty, learning_rate, num_iterations):
    feature_parameters = get_feature_parameters(feature)

    # The headers of the CSV file are defined as the absolute path and transcription of each and every command file.
    command_table = read_csv(csv_path=command_csv)
    integrated_command_table = [["audio path", "transcription", "loss_feature", "loss_delta", "loss", "SNR"]]

    song_filenames = os.listdir(song_dir)
    song_filenames.sort()

    for command_info in command_table[1: ]:
        command_name = os.path.basename(command_info[0])[: -4]
        integrated_command_dir = os.path.join(dir, "0", "{}-{}".format(command_name, feature))
        song_clip_dir = os.path.join(dir, "1", "{}-{}".format(command_name, feature))

        if not os.path.exists(integrated_command_dir):
            os.makedirs(integrated_command_dir)
        if not os.path.exists(song_clip_dir):
            os.makedirs(song_clip_dir)
        
        command, _ = load(filepath=command_info[0])

        for feature_parameters_dict in feature_parameters:
            feature_extractor = get_feature_extractor(feature, feature_parameters_dict)

            for song_filename in song_filenames:
                song_path = os.path.join(song_dir, song_filename)
                song, _ = load(filepath=song_path)

                for origin in range(0, song.size()[1] - command.size()[1] + 1, int(16000 * interval)):
                    time_origin = format(origin / 16000, '.1f')
                    integrated_command_filename = "{}-{}-{}-{}-{}-{}-{}-{}-{}-{}.wav".format(command_name, song_filename, time_origin, 
                                                                                             feature_parameters_dict.get("n_mfcc", ""), 
                                                                                             feature_parameters_dict.get("n_mels", ""), 
                                                                                             feature_parameters_dict.get("n_fft", ""), 
                                                                                             optimizer, penalty, learning_rate, num_iterations)
                    integrated_command_path = os.path.join(integrated_command_dir, integrated_command_filename)
                    song_clip_path = os.path.join(song_clip_dir, integrated_command_filename)
                    
                    logging.info("Start to generate {}".format(integrated_command_path))

                    loss_feature, loss_delta, loss, snr = attack_sample(feature_extractor, command_info[0], song_path, integrated_command_path, song_clip_path, origin, 
                                                                        penalty, optimizer, learning_rate, num_iterations)

                    integrated_command_table.append([integrated_command_path, command_info[1], loss_feature, loss_delta, loss, snr])
    
    write_csv(csv_path=os.path.join(dir, "integrated-commands-{}.csv".format(feature)), lines=integrated_command_table)


def get_feature_extractor(feature, feature_parameters_dict):
    if feature == 1:
        return Spectrogram(n_fft=feature_parameters_dict["n_fft"]).to(DEVICE)
    elif feature == 2:
        return MelSpectrogram(n_mels=feature_parameters_dict["n_mels"], 
                              n_fft=feature_parameters_dict["n_fft"]).to(DEVICE)
    else:
        return MFCC(n_mfcc=feature_parameters_dict["n_mfcc"], 
                    melkwargs={"n_mels": feature_parameters_dict["n_mels"], "n_fft": feature_parameters_dict["n_fft"]}).to(DEVICE)


def attack_sample(feature_extractor, command_path, song_path, integrated_command_path, song_clip_path, origin, penalty, optimizer, learning_rate, num_iterations):
    command, _ = load(filepath=command_path)
    song, _ = load(filepath=song_path)

    # Intercept a song snippet of the same length as the command.
    song = song[:, origin: origin + command.size()[1]]

    command = command.to(DEVICE)
    song = song.to(DEVICE)

    # Initialize the perturbation vector.
    delta = Variable(torch.zeros(size=song.size()).to(DEVICE), requires_grad=True)
    command_feature = feature_extractor(waveform=command)
    integrated_command_feature = feature_extractor(waveform=song+delta)

    loss_feature = torch.norm(integrated_command_feature - command_feature)
    loss_delta = torch.norm(delta)
    loss = loss_feature + penalty * loss_delta
    
    logging.info("loss_feature = {:.4f}, loss_delta = {:.4f}, loss = {:.4f}".format(loss_feature.data, loss_delta.data, loss.data))

    # Choose an optimizer. You can add more choices if needed.
    if optimizer == "Adam":
        optimizer = Adam(params=[delta], lr=learning_rate)
    elif optimizer == "SGD":
        optimizer = SGD(params=[delta], lr=learning_rate)
    else:
        logging.error("Only support Adam and SGD!")
        return
    
    loss_feature_trend = torch.zeros(num_iterations)
    loss_delta_trend = torch.zeros(num_iterations)
    loss_trend = torch.zeros(num_iterations)
    
    # Minimize the objective: ||integrated_command_feature - command_feature|| + penalty * ||delta||
    for i in range(num_iterations):
        integrated_command_feature = feature_extractor(waveform=song+delta)

        loss_feature = torch.norm(integrated_command_feature - command_feature)
        loss_delta = torch.norm(delta)
        loss = loss_feature + penalty * loss_delta
        
        loss_feature_trend[i] = loss_feature
        loss_delta_trend[i] = loss_delta
        loss_trend[i] = loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    integrated_command = song + delta
    integrated_command = integrated_command.clamp(min=-1.0, max=1.0).detach().cpu()
    song_clip = song.detach().cpu()

    snr = get_snr(audio=song_clip, audio_with_noise=integrated_command)

    save(filepath=integrated_command_path, src=integrated_command, sample_rate=16000)
    save(filepath=song_clip_path, src=song_clip, sample_rate=16000)

    logging.info("loss_feature = {:.4f}, loss_delta = {:.4f}, loss = {:.4f}, snr = {:.2f}".format(loss_feature.data, loss_delta.data, loss.data, snr))
    logging.info("")

    fig, ax = plt.subplots(nrows=1, ncols=3, sharex='row')
    ax[0].plot(loss_feature_trend.detach().cpu())
    ax[1].plot(loss_delta_trend.detach().cpu())
    ax[2].plot(loss_trend.detach().cpu())
    fig.savefig("{}.png".format(integrated_command_path))

    return format(loss_feature.data, '.4f'), format(loss_delta.data, '.4f'), format(loss, '.4f'), format(snr, '.2f')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-f", "--feature", type=int, default=1, help="")
    parser.add_argument("-c", "--command_csv", type=str, default="commands/commands.csv", help="")
    parser.add_argument("-s", "--song_dir", type=str, default="songs/", help="")
    parser.add_argument("-d", "--dir", type=str, default="integrated-commands/", help="")
    parser.add_argument("-i", "--interval", type=float, default=1, help="Time interval to intercept a song.")
    parser.add_argument("-p", "--penalty", type=float, default=500, help="Weight of the norm of the malicious perturbation.")
    parser.add_argument("-o", "--optimizer", type=str, default="Adam", help="Integrated optimizers in PyTorch, including Adam and SGD.")
    parser.add_argument("-l", "--learning_rate", type=float, default=0.001, help="Learning rate used in the specified optimizer.")
    parser.add_argument("-n", "--num_iterations", type=int, default=3000, help="The maximum number of iterations for the specified optimizer.")
    parser.add_argument("-g", "--gpu", type=str, default='0', help="GPU index to use.")

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    integrated_command_attack(feature=args.feature, command_csv=args.command_csv, song_dir=args.song_dir, dir=args.dir, interval=args.interval, 
                              optimizer=args.optimizer, penalty=args.penalty, learning_rate=args.learning_rate, num_iterations=args.num_iterations)