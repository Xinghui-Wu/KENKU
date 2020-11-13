import argparse
import os
import random

import torch
from torch.autograd import Variable
from torch.optim import Adam, SGD
from torchaudio import load, save
from torchaudio.transforms import MFCC


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MFCC_CALCULATOR = MFCC().to(DEVICE)


def attack_dataset(commands, songs, malicious_samples, optimizer, penalty_factor, learning_rate, num_iterations):
    """[summary]

    Args:
        commands (str): [description]
        songs (str): [description]
        malicious_samples (str): [description]
        optimizer (str): [description]
        penalty_factor (float): [description]
        learning_rate (float): [description]
        num_iterations (int): [description]
    """
    command_filenames = os.listdir(commands)
    command_filenames.sort()

    song_filenames = os.listdir(songs)
    song_filenames.sort()

    for i in range(len(command_filenames)):
        command_path = os.path.join(commands, command_filenames[i])

        for j in range(len(song_filenames)):
            print("****************************************************************************************************")
            print(command_filenames[i], song_filenames[j])
            print()

            song_path = os.path.join(songs, song_filenames[j])
            malicious_sample_path = os.path.join(malicious_samples, "Malicious-Sample-{}-{}.wav".format(i, j))

            attack_sample(command_path, song_path, malicious_sample_path, optimizer, penalty_factor, learning_rate, num_iterations)

            print("****************************************************************************************************")
            print()


def attack_sample(command_path, song_path, malicious_sample_path, optimizer, penalty_factor, learning_rate, num_iterations):
    """[summary]

    Args:
        command_path ([type]): [description]
        song_path ([type]): [description]
        malicious_sample_path ([type]): [description]
        optimizer ([type]): [description]
        penalty_factor ([type]): [description]
        learning_rate ([type]): [description]
        num_iterations ([type]): [description]
    """
    command, sr = load(command_path)
    song, sr = load(song_path)

    origin = random.randint(80000, song.size()[1] - command.size()[1] - 80000)
    song = song[:, origin: origin + command.size()[1]]

    command = command.to(DEVICE)
    song = song.to(DEVICE)

    command_mfcc = MFCC_CALCULATOR.forward(waveform=command)

    delta = Variable((0.02 * (torch.rand(size=song.size()) - 0.5)).to(DEVICE), requires_grad=True)
    malicious_sample_mfcc = MFCC_CALCULATOR.forward(waveform=song+delta)
    
    print("By contrast:")
    print("||0.1 * song|| = {:.4f}".format(torch.norm(0.1 * song).data))
    print("||0.1 * command_mfcc|| = {:.4f}".format(torch.norm(0.1 * command_mfcc).data))
    print()

    print("At the beginning of the attack:")
    print("||delta|| = {:.4f}".format(torch.norm(delta).data))
    print("||malicious_sample_mfcc - command_mfcc|| = {:.4f}".format(torch.norm(malicious_sample_mfcc - command_mfcc).data))
    print()

    if optimizer == "Adam":
        optimizer = Adam(params=[delta], lr=learning_rate)
    elif optimizer == "SGD":
        optimizer = SGD(params=[delta], lr=learning_rate)
    else:
        print("Only support Adam and SGD!")
        return

    for i in range(num_iterations):
        optimizer.zero_grad()
        malicious_sample_mfcc = MFCC_CALCULATOR.forward(waveform=song+delta)
        loss = torch.norm(malicious_sample_mfcc - command_mfcc) + penalty_factor * torch.norm(delta)
        loss.backward()
        optimizer.step()

    malicious_sample = song + delta
    malicious_sample = malicious_sample.clamp(min=-1.0, max=1.0)
    malicious_sample = malicious_sample.detach().cpu()

    print("At the end of the attack:")
    print("||delta|| = {:.4f}".format(torch.norm(delta).data))
    print("||malicious_sample_mfcc - command_mfcc|| = {:.4f}".format(torch.norm(malicious_sample_mfcc - command_mfcc).data))

    save(filepath=malicious_sample_path, src=malicious_sample, sample_rate=16000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-c", "--commands", type=str, default="./Audio Samples/Commands/", help="")
    parser.add_argument("-s", "--songs", type=str, default="./Audio Samples/Songs/", help="")
    parser.add_argument("-m", "--malicious_samples", type=str, default="./Audio Samples/Malicious Samples/", help="")
    parser.add_argument("-o", "--optimizer", type=str, default="Adam", help="")
    parser.add_argument("-p", "--penalty_factor", type=float, default=25, help="")
    parser.add_argument("-l", "--learning_rate", type=float, default=0.001, help="")
    parser.add_argument("-n", "--num_iterations", type=int, default=10000, help="")
    args = parser.parse_args()

    attack_dataset(commands=args.commands, songs=args.songs, malicious_samples=args.malicious_samples, 
                   optimizer=args.optimizer, penalty_factor=args.penalty_factor, learning_rate=args.learning_rate, num_iterations=args.num_iterations)
