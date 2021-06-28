import argparse
import os

import torch
from torch.autograd import Variable
from torch.optim import Adam, SGD
from torchaudio import load, save
from torchaudio.transforms import MFCC


def attack_dataset(commands, songs, pure_samples, malicious_samples, transcriptions, interval, optimizer, penalty_factor, learning_rate, num_iterations):
    """Inject commands into songs and generate malicious samples to attack ASR systems.
    The proposed attack method can be abstracted into an optimization model.
    This script uses the integrated optimizers in PyTorch and adjust some hyperparameters, including the penalty factor, learning rate and the number of iterations, to solve the problem.
    This method accepts a input command dataset and a input song dataset.
    Inject each of the commands into each of the songs one by one to generate output datasets of pure samples and malicious samples respectively.
    This method will iterate all command files and all song files and call attack_sample method below at a time.
    Finally, output a transcription file of the corresponding malicious samples to the same directory of the transcription file of the pure commands.

    Args:
        commands (str): Input directory of some command files.
        songs (str): Input directory of some song files.
        pure_samples (str): Output directory of the generated pure samples.
        malicious_samples (str): Output directory of the generated malicious samples.
        transcriptions (str): Path of the transcription file of the commands.
        interval (float): Time interval to intercept a song.
        optimizer (str): Integrated optimizers in PyTorch, including Adam and SGD.
        penalty_factor (float): Weight of the norm of the malicious perturbation.
        learning_rate (float): Learning rate used in the specified optimizer.
        num_iterations (int): The maximum number of iterations for the specified optimizer.
    """
    command_filenames = os.listdir(commands)
    command_filenames.sort()

    song_filenames = os.listdir(songs)
    song_filenames.sort()

    with open(transcriptions, 'r') as transcription_txt:
        command_transcriptions = transcription_txt.readlines()
    
    malicious_transcriptions = list()

    for i in range(len(command_filenames)):
        command_path = os.path.join(commands, command_filenames[i])
        command, sr = load(filepath=command_path)

        for j in range(len(song_filenames)):
            song_path = os.path.join(songs, song_filenames[j])
            song, sr = load(filepath=song_path)

            for origin in range(0, song.size()[1] - command.size()[1] + 1, int(16000 * interval)):
                time_origin = format(origin / 16000, '.1f')

                print("****************************************************************************************************")
                print(command_filenames[i], song_filenames[j], time_origin)
                print()
                
                pure_sample_path = os.path.join(pure_samples, "Pure-Sample-{}-{}-{}.wav".format(i, j, time_origin))
                malicious_sample_path = os.path.join(malicious_samples, "Malicious-Sample-{}-{}-{}.wav".format(i, j, time_origin))
                
                attack_sample(command_path, song_path, pure_sample_path, malicious_sample_path, origin, 
                              optimizer, penalty_factor, learning_rate, num_iterations)
                
                print("****************************************************************************************************")
                print()

                malicious_transcriptions.append(command_transcriptions[i])
    
    with open(os.path.join(os.path.dirname(transcriptions), "Malicious-Commands.txt"), 'w') as malicious_transcriptions_txt:
        malicious_transcriptions_txt.writelines(malicious_transcriptions)
    
    with open(os.path.join(os.path.dirname(transcriptions), "parameters.txt"), 'w') as parameters_txt:
        parameters_txt.writelines(["interval = {}\n".format(interval), 
                                   "optimizer = {}\n".format(optimizer), 
                                   "penalty_factor = {}\n".format(penalty_factor), 
                                   "learning_rate = {}\n".format(learning_rate), 
                                   "num_iterations = {}\n".format(num_iterations)])


def attack_sample(command_path, song_path, pure_sample_path, malicious_sample_path, origin, optimizer, penalty_factor, learning_rate, num_iterations):
    """Inject a command into a song clip and generate a malicious sample to attack ASR systems.
    The basic idea is to craft a song slightly so that the MFCC of the crafted song clip can approach the MFCC of the desired command.
    The proposed attack method can be abstracted into an optimization model.
    This script uses the integrated optimizers in PyTorch and adjust some hyperparameters, including the penalty factor, learning rate and the number of iterations, to solve the problem.
    This method accepts a input command file and a input song file and the clip origin.
    Inject the command into the song clip to generate the output pure sample and malicious sample respectively.

    Args:
        command_path (str): Input path of a command file.
        song_path (str): Input path of a song file.
        pure_sample_path (str): Output path of the generated pure sample.
        malicious_sample_path (str): Output path of the generated malicious sample.
        origin (float): Origin of the sampling point to intercept the song.
        optimizer (str): Integrated optimizers in PyTorch, including Adam and SGD.
        penalty_factor (float): Weight of the norm of the malicious perturbation.
        learning_rate (float): Learning rate used in the specified optimizer.
        num_iterations (int): The maximum number of iterations for the specified optimizer.
    """
    command, _ = load(filepath=command_path)
    song, _ = load(filepath=song_path)

    # Intercept a song snippet of the same length as the command.
    song = song[:, origin: origin + command.size()[1]]

    command = command.to(DEVICE)
    song = song.to(DEVICE)

    command_mfcc = MFCC_CALCULATOR.forward(waveform=command)

    # Initialize the perturbation vector.
    delta = Variable((0.02 * (torch.rand(size=song.size()) - 0.5)).to(DEVICE), requires_grad=True)
    malicious_sample_mfcc = MFCC_CALCULATOR.forward(waveform=song+delta)

    # Choose an optimizer. You can add more choices if needed.
    if optimizer == "Adam":
        optimizer = Adam(params=[delta], lr=learning_rate)
    elif optimizer == "SGD":
        optimizer = SGD(params=[delta], lr=learning_rate)
    else:
        print("Only support Adam and SGD!")
        return
    
    print("At the beginning of the attack:")
    print("||delta|| = {:.4f}".format(torch.norm(delta).data))
    print("||malicious_sample_mfcc - command_mfcc|| = {:.4f}".format(torch.norm(malicious_sample_mfcc - command_mfcc).data))
    print()

    # Minimize the objective: ||malicious_sample_mfcc - command_mfcc|| + penalty_factor * ||delta||
    for i in range(num_iterations):
        optimizer.zero_grad()
        malicious_sample_mfcc = MFCC_CALCULATOR.forward(waveform=song+delta)
        loss = torch.norm(malicious_sample_mfcc - command_mfcc) + penalty_factor * torch.norm(delta)
        loss.backward()
        optimizer.step()
    
    print("At the end of the attack:")
    print("||delta|| = {:.4f}".format(torch.norm(delta).data))
    print("||malicious_sample_mfcc - command_mfcc|| = {:.4f}".format(torch.norm(malicious_sample_mfcc - command_mfcc).data))

    pure_sample = song.detach().cpu()

    malicious_sample = song + delta
    malicious_sample = malicious_sample.clamp(min=-1.0, max=1.0)
    malicious_sample = malicious_sample.detach().cpu()

    save(filepath=pure_sample_path, src=pure_sample, sample_rate=16000)
    save(filepath=malicious_sample_path, src=malicious_sample, sample_rate=16000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inject commands into songs and generate malicious samples to attack ASR systems.")
    parser.add_argument("-c", "--commands", type=str, default="./Audio Samples/Commands/", help="Input directory of some command files.")
    parser.add_argument("-s", "--songs", type=str, default="./Audio Samples/Carriers/", help="Input directory of some song files.")
    parser.add_argument("-p", "--pure_samples", type=str, default="./Audio Samples/Pure Samples/", help="Output directory of the generated pure samples.")
    parser.add_argument("-m", "--malicious_samples", type=str, default="./Audio Samples/Malicious Samples/", help="Output directory of the generated malicious samples.")
    parser.add_argument("-t", "--transcriptions", type=str, default="./Audio Samples/Commands.txt", help="Path of the transcription file of the commands.")
    parser.add_argument("-i", "--interval", type=float, default=1, help="Time interval to intercept a song.")
    parser.add_argument("-o", "--optimizer", type=str, default="Adam", help="Integrated optimizers in PyTorch, including Adam and SGD.")
    parser.add_argument("-f", "--penalty_factor", type=float, default=75, help="Weight of the norm of the malicious perturbation.")
    parser.add_argument("-l", "--learning_rate", type=float, default=0.001, help="Learning rate used in the specified optimizer.")
    parser.add_argument("-n", "--num_iterations", type=int, default=10000, help="The maximum number of iterations for the specified optimizer.")
    parser.add_argument("-g", "--gpu", type=str, default='0', help="GPU index to use.")
    parser.add_argument("--n_mfcc", type=int, default=40, help="The MFCC parameter to decide the number of the coefficients.")
    
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    MFCC_CALCULATOR = MFCC(args.n_mfcc).to(DEVICE)

    attack_dataset(commands=args.commands, songs=args.songs, 
                   pure_samples=args.pure_samples, malicious_samples=args.malicious_samples, 
                   transcriptions=args.transcriptions, interval=args.interval, 
                   optimizer=args.optimizer, penalty_factor=args.penalty_factor, 
                   learning_rate=args.learning_rate, num_iterations=args.num_iterations)
