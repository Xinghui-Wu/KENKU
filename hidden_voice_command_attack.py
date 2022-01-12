import argparse
import os
import logging

import matplotlib.pyplot as plt
import torch
from torch.tensor import Tensor
from torch.autograd import Variable
from torch.optim import Adam, SGD
from torchaudio import load, save
from torchaudio.transforms import MFCC, MelSpectrogram, Spectrogram, AmplitudeToDB

from utils import *

logging.basicConfig(level=logging.INFO, format="%(asctime)s \t %(message)s", datefmt="%Y-%m-%d %H:%M:%S")


class DBSpectrogram(Spectrogram):
    def __init__(self, n_fft, hop_length):
        super().__init__(n_fft=n_fft, hop_length=hop_length)
    
    def forward(self, waveform: Tensor) -> Tensor:
        return AMPLITUDE_TO_DB(super().forward(waveform))


def hidden_voice_command_attack(feature, command_csv, dir, optimizer, penalty, learning_rate, num_iterations):
    feature_parameters = get_feature_parameters(feature)

    # The headers of the CSV file are defined as the absolute path and transcription of each and every command file.
    command_table = read_csv(csv_path=command_csv)
    hidden_voice_command_table = [["audio path", "transcription", "loss_feature", "loss_perturbation", "loss", "SNR"]]

    hidden_voice_command_dir = os.path.join(dir, "0", "{}".format(feature))
    perturbation_dir = os.path.join(dir, "2", "{}".format(feature))

    if not os.path.exists(hidden_voice_command_dir):
        os.makedirs(hidden_voice_command_dir)
    if not os.path.exists(perturbation_dir):
        os.makedirs(perturbation_dir)
    
    for command_info in command_table[1: ]:
        command_name = os.path.basename(command_info[0])[: -4]
        
        for feature_parameters_dict in feature_parameters:
            feature_extractor = get_feature_extractor(feature, feature_parameters_dict)

            hidden_voice_command_filename = "{}-{}-{}-{}-{}-{}-{}-{}-{}.wav".format(command_name, 
                                                                                    feature_parameters_dict.get("n_mfcc", ""), 
                                                                                    feature_parameters_dict.get("n_mels", ""), 
                                                                                    feature_parameters_dict.get("n_fft"), 
                                                                                    feature_parameters_dict.get("hop_length"), 
                                                                                    optimizer, penalty, learning_rate, num_iterations)
            hidden_voice_command_path = os.path.join(hidden_voice_command_dir, hidden_voice_command_filename)
            perturbation_path = os.path.join(perturbation_dir, hidden_voice_command_filename)

            logging.info("Start to generate {}".format(hidden_voice_command_path))

            loss_feature, loss_perturbation, loss, snr = attack_sample(feature_extractor, command_info[0], hidden_voice_command_path, perturbation_path, 
                                                                       penalty, optimizer, learning_rate, num_iterations)

            hidden_voice_command_table.append([hidden_voice_command_path, command_info[1], loss_feature, loss_perturbation, loss, snr])

    write_csv(csv_path=os.path.join(dir, "hidden-voice-commands-{}.csv".format(feature)), lines=hidden_voice_command_table)


def get_feature_extractor(feature, feature_parameters_dict):
    if feature == 0:
        return DBSpectrogram(n_fft=feature_parameters_dict["n_fft"], hop_length=feature_parameters_dict.get("hop_length")).to(DEVICE)
    elif feature == 1:
        return Spectrogram(n_fft=feature_parameters_dict["n_fft"], hop_length=feature_parameters_dict.get("hop_length")).to(DEVICE)
    elif feature == 2:
        return MelSpectrogram(n_mels=feature_parameters_dict["n_mels"], 
                              n_fft=feature_parameters_dict["n_fft"], 
                              hop_length=feature_parameters_dict.get("hop_length")).to(DEVICE)
    else:
        return MFCC(n_mfcc=feature_parameters_dict["n_mfcc"], 
                    melkwargs={"n_mels": feature_parameters_dict["n_mels"], 
                               "n_fft": feature_parameters_dict["n_fft"], 
                               "hop_length": feature_parameters_dict.get("hop_length")}).to(DEVICE)


def attack_sample(feature_extractor, command_path, hidden_voice_command_path, perturbation_path, penalty, optimizer, learning_rate, num_iterations):
    command, _ = load(filepath=command_path)
    command = command.to(DEVICE)

    # Initialize the perturbation vector.
    perturbation = Variable((0.0002 * (torch.rand(size=command.size()) - 0.5)).to(DEVICE), requires_grad=True)

    # Choose an optimizer. You can add more choices if needed.
    if optimizer == "Adam":
        optimizer = Adam(params=[perturbation], lr=learning_rate)
    elif optimizer == "SGD":
        optimizer = SGD(params=[perturbation], lr=learning_rate)
    else:
        logging.error("Only support Adam and SGD!")
        return
    
    command_feature = feature_extractor(command)
    hidden_voice_command_feature = feature_extractor(command + perturbation)

    loss_feature = torch.norm(hidden_voice_command_feature - command_feature)
    loss_perturbation = torch.norm(perturbation)
    loss = loss_feature - penalty * loss_perturbation

    logging.info("loss_feature = {:.4f}, loss_perturbation = {:.4f}, loss = {:.4f}".format(loss_feature.data, loss_perturbation.data, loss.data))
    
    loss_feature_trend = torch.zeros(num_iterations)
    loss_perturbation_trend = torch.zeros(num_iterations)
    loss_trend = torch.zeros(num_iterations)

    # Minimize the objective: loss = loss_feature + penalty * loss_perturbation
    for i in range(num_iterations):
        hidden_voice_command_feature = feature_extractor(command + perturbation)

        loss_feature = torch.norm(hidden_voice_command_feature - command_feature)
        loss_perturbation = torch.norm(perturbation)
        loss = loss_feature - penalty * loss_perturbation
        
        loss_feature_trend[i] = loss_feature
        loss_perturbation_trend[i] = loss_perturbation
        loss_trend[i] = loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    hidden_voice_command = command + perturbation
    hidden_voice_command = hidden_voice_command.clamp(min=-1.0, max=1.0).detach().cpu()
    perturbation = perturbation.detach().cpu()

    snr = get_snr(audio=command.detach().cpu().numpy(), audio_with_noise=hidden_voice_command.numpy())

    save(filepath=hidden_voice_command_path, src=hidden_voice_command, sample_rate=16000)
    save(filepath=perturbation_path, src=perturbation, sample_rate=16000)

    logging.info("loss_feature = {:.4f}, loss_perturbation = {:.4f}, loss = {:.4f}, snr = {:.2f}".format(loss_feature.data, loss_perturbation.data, loss.data, snr))
    logging.info("")

    fig, ax = plt.subplots(nrows=1, ncols=3, sharex='row')
    ax[0].plot(loss_feature_trend.detach().cpu().numpy())
    ax[1].plot(loss_perturbation_trend.detach().cpu().numpy())
    ax[2].plot(loss_trend.detach().cpu().numpy())
    fig.savefig("{}.png".format(hidden_voice_command_path))

    return format(loss_feature.data, '.4f'), format(loss_perturbation.data, '.4f'), format(loss, '.4f'), format(snr, '.2f')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-f", "--feature", type=int, default=3, help="")
    parser.add_argument("-c", "--command_csv", type=str, default="commands/commands.csv", help="")
    parser.add_argument("-d", "--dir", type=str, default="hidden-voice-commands/", help="")
    parser.add_argument("-p", "--penalty", type=float, default=1000, help="Weight of the norm of the malicious perturbation.")
    parser.add_argument("-o", "--optimizer", type=str, default="Adam", help="Integrated optimizers in PyTorch, including Adam and SGD.")
    parser.add_argument("-l", "--learning_rate", type=float, default=0.0001, help="Learning rate used in the specified optimizer.")
    parser.add_argument("-n", "--num_iterations", type=int, default=10000, help="The maximum number of iterations for the specified optimizer.")
    parser.add_argument("-g", "--gpu", type=str, default='0', help="GPU index to use.")

    args = parser.parse_args()

    if args.feature == 0:
        AMPLITUDE_TO_DB = AmplitudeToDB()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    hidden_voice_command_attack(feature=args.feature, command_csv=args.command_csv, dir=args.dir, 
                                optimizer=args.optimizer, penalty=args.penalty, learning_rate=args.learning_rate, num_iterations=args.num_iterations)
