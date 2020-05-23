import argparse
import torch
from torch.autograd import Variable
from torchaudio import load, save
from torchaudio.transforms import MFCC
from tqdm import tqdm


def main(args):
    # If CUDA is available, use CUDA to speed up calculations.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Instantiate a MFCC calculator.
    mfcc_calculator = MFCC()
    mfcc_calculator.to(device)

    song_path = "./Audio Samples/Songs/Song-{}.wav".format(args.song_index)
    command_path = "./Audio Samples/Commands/Command-{}.wav".format(args.command_index)
    output_path = "./Audio Samples/Malicious Samples/{}-{}-3-{}-{}-{}.wav".format(
                  args.song_index, args.command_index, args.init_delta_range, args.iterations, args.lr)

    attack(device, mfcc_calculator, song_path, command_path, output_path,
           origin=args.origin, init_delta_range=args.init_delta_range, iterations=args.iterations, lr=args.lr)


def attack(device, mfcc_calculator, song_path, command_path, output_path, origin, init_delta_range, iterations, lr):
    """An attack method based on gradient descent with PyTorch.

    Arguments:
        device {torch.device} -- CUDA or CPU
        mfcc_calculator {torchaudio.transforms.MFCC} -- MFCC calculator
        song_path {str} -- path of the song file
        command_path {str} -- path of the command file
        output_path {str} -- path of the attack audio file
        origin {float} -- origin of the song to inject the command
        init_delta_range {float} -- initial delta is in (-0.5 * init_delta_range, 0.5 * init_delta_range)
        iterations {int} -- number of iterations of gradient descent
        lr {float} -- learning rate
    """
    song, sr = load(song_path)
    command, sr = load(command_path)

    # Intercept song fragments as the same length of the target command.
    song = song[:, int(origin * sr): int(origin * sr) + command.size()[1]]
    song = song.to(device)
    command = command.to(device)
    
    command_mfcc = mfcc_calculator.forward(waveform=command)

    delta = Variable((init_delta_range * (torch.rand(size=song.size()) - 0.5)).to(device), requires_grad=True)
    print(torch.norm(delta))
    pbar = tqdm(total=iterations)
    for i in range(iterations):
        attack_song = song + delta
        attack_song_mfcc = mfcc_calculator.forward(waveform=attack_song)
        loss = torch.norm(attack_song_mfcc - command_mfcc)
        loss.backward()
        delta.data = delta.data - lr * delta.grad.data
        delta.grad.data.zero_()
        pbar.update(1)
    pbar.close()
    
    print(torch.norm(delta))
    print(torch.norm(0.01 * song))
    print(torch.norm(attack_song_mfcc - command_mfcc))
    print(torch.norm(0.01 * command_mfcc))

    save(filepath=output_path, src=attack_song.detach().cpu(), sample_rate=sr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="An attack method based on gradient descent with PyTorch")
    parser.add_argument("--song_index", type=int, default=1, help="choose a song based on its index")
    parser.add_argument("--command_index", type=int, default=1, help="choose a command based on its index")
    parser.add_argument("--origin", type=float, default=1.0, help="origin of the song to inject the command")
    parser.add_argument("--init_delta_range", type=float, default=0.002, help="initial delta is in (-0.5 * init_delta_range, 0.5 * init_delta_range)")
    parser.add_argument("--iterations", type=int, default=100, help="number of iterations of gradient descent")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    arguments = parser.parse_args()
    main(arguments)
