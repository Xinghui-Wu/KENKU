import argparse
from librosa import load
from librosa.feature import mfcc
from librosa.feature.inverse import mfcc_to_audio
from librosa.output import write_wav

from utils import convert


def main(args):
    song_path = "./Audio Samples/Songs/Song-{}.wav".format(args.song_index)
    command_path = "./Audio Samples/Commands/Command-{}.wav".format(args.command_index)
    output_path = "./Audio Samples/Malicious Samples/{}-{}-2-{}.wav".format(
                  args.song_index, args.command_index, args.iterations)

    attack(song_path, command_path, output_path, iterations=args.iterations)


def attack(song_path, command_path, output_path, iterations):
    sr = 16000
    song, _ = load(path=song_path, sr=sr)
    command, _ = load(path=command_path, sr=sr)

    for i in range(iterations):
        command_mfcc = mfcc(y=command, sr=sr)
        command = mfcc_to_audio(mfcc=command_mfcc)
    write_wav(path=output_path, y=command, sr=sr)
    convert(path=output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="An attack method based on Griffin-Lim to reverse MFCC")
    parser.add_argument("--song_index", type=int, default=1, help="choose a song based on its index")
    parser.add_argument("--command_index", type=int, default=1, help="choose a command based on its index")
    parser.add_argument("--iterations", type=int, default=5, help="number of iterations of inverse MFCC")
    arguments = parser.parse_args()
    main(arguments)
