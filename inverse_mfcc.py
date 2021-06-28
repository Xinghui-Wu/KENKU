import argparse
import os

import numpy as np
from librosa import load
from librosa.feature import mfcc
from librosa.feature.inverse import mfcc_to_audio
from soundfile import write


def attack(commands, malicious_samples, transcriptions, num_iterations):
    """Reverse the MFCC feature of an audio iteratively based on the Griffin-Lim algorithm.
    For each command, get its MFCC and try to reverse it back to a time-domain signal for multiple times.
    This script automatically test the parameters involved in the MFCC feature extraction procedure.

    Args:
        commands (str): Input directory of some command files.
        malicious_samples (str): Output directory of the generated malicious samples.
        transcriptions (str): Path of the transcription file of the commands.
        num_iterations (int): The number of iterations to reverse MFCC.
    """
    command_filenames = os.listdir(commands)
    command_filenames.sort()

    with open(transcriptions, 'r') as transcription_txt:
        command_transcriptions = transcription_txt.readlines()
    
    malicious_transcriptions = list()

    for n_mfcc in [12, 13, 20, 26, 40]:
        for n_mels in [64, 128, 256, 512]:
            for i in range(len(command_filenames)):
                print("****************************************************************************************************")
        
                command_path = os.path.join(commands, command_filenames[i])
                command, _ = load(path=command_path, sr=16000)
                target_command_mfcc = mfcc(y=command, sr=16000, n_mfcc=n_mfcc)

                for j in range(num_iterations):
                    print(command_filenames[i], j, n_mfcc, n_mels)

                    command_mfcc = mfcc(y=command, sr=16000, n_mfcc=n_mfcc)
                    command = mfcc_to_audio(mfcc=command_mfcc, n_mels=n_mels)

                    print("||malicious_sample_mfcc - command_mfcc|| = {:.4f}".format(np.linalg.norm(mfcc(y=command, sr=16000, n_mfcc=n_mfcc) - target_command_mfcc)))

                    malicious_sample_path = os.path.join(malicious_samples, "Malicious-Sample-{}-{}-{}-{}.wav".format(i, j, n_mfcc, n_mels))
                    write(file=malicious_sample_path, data=command, samplerate=16000)
        
                print("****************************************************************************************************")
                print()
    
    for command_transcription in command_transcriptions:
        for i in range(5 * 4 * num_iterations):
            malicious_transcriptions.append(command_transcription)
    
    with open(os.path.join(os.path.dirname(transcriptions), "Malicious-Commands.txt"), 'w') as malicious_transcriptions_txt:
        malicious_transcriptions_txt.writelines(malicious_transcriptions)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Reverse the MFCC feature of an audio iteratively based on the Griffin-Lim algorithm.")
    parser.add_argument("-c", "--commands", type=str, default="./Audio Samples/Commands/", help="Input directory of some command files.")
    parser.add_argument("-m", "--malicious_samples", type=str, default="./Audio Samples/Malicious Samples/", help="Output directory of the generated malicious samples.")
    parser.add_argument("-t", "--transcriptions", type=str, default="./Audio Samples/Commands.txt", help="Path of the transcription file of the commands.")
    parser.add_argument("-n", "--num_iterations", type=int, default=3, help="The number of iterations to reverse the MFCC.")
    
    args = parser.parse_args()
    
    attack(commands=args.commands, malicious_samples=args.malicious_samples, 
           transcriptions=args.transcriptions, num_iterations=args.num_iterations)
