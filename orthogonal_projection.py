import argparse
import numpy as np
import math
import random
import scipy
from librosa import load
from librosa.output import write_wav

from algebraic_mfcc import algebraic_mfcc
from utils import convert


def main(args):
    song_path = "./Audio Samples/Songs/Song-{}.wav".format(args.song_index)
    command_path = "./Audio Samples/Commands/Command-{}.wav".format(args.command_index)
    output_path = "./Audio Samples/Malicious Samples/{}-{}-1.wav".format(args.song_index, args.command_index)

    attack(song_path, command_path, output_path, origin=args.origin)


def attack(song_path, command_path, output_path, origin):
    """An attack method based on orthogonal projection.

    Arguments:
        song_path {str} -- path of the song file
        command_path {str} -- path of the command file
        output_path {str} -- path of the attack audio file
        origin {float} -- origin of the song to inject the command
    """
    sr = 16000
    song, _ = load(song_path, sr=sr)
    command, _ = load(command_path, sr=sr)
    
    # Intercept song fragments as the same length of the target command.
    song = song[int(origin * sr): int(origin * sr) + len(command)]

    # Extract MFCCs of the song and the command respectively as well as each constant matrix.
    A, C, D, E, F, F1, song_MFCC = algebraic_mfcc(x=song, sr=sr)
    A, C, D, E, F, F1, command_MFCC = algebraic_mfcc(x=command, sr=sr)

    # Let G=F1*F, according to the combination law of matrix multiplication.
    G = np.matmul(F1, F)
    # Get the standard orthogonal basis of the subspace of column vectors of G.
    orth_G = scipy.linalg.orth(G)
    # Project each column vector of matrix command_MFCC orthogonally to the subspace of column vectors of G.
    projection = np.matmul(orth_G.T, command_MFCC)

    # generalized inverse matrix of G
    pinv_G = np.linalg.pinv(G)
    X = np.matmul(pinv_G, projection)
    X = np.exp(X)

    # generalized inverse matrix of E
    pinv_E = np.linalg.pinv(E)
    X = np.matmul(pinv_E, X)

    # Randomly disassemble the sum of squares.
    X = len(X) * X
    K, N_frames = X.shape
    cX = np.zeros(shape=X.shape, dtype='complex64')
    for i in range(K):
        for j in range(N_frames):
            if X[i][j] > 0:
                a = random.uniform(0, X[i][j])
                a = math.sqrt(a)
                b = math.sqrt(X[i][j] - a ** 2)
                a = a if random.random() >= 0.5 else -a
                b = b if random.random() >= 0.5 else -b
                cX[i][j] = complex(a, b)
            else:
                cX[i][j] = complex(0, 0)
    
    # inverse transform matrix of DFT
    N = D.shape[1]
    inverse_D = np.zeros(shape=(N, K), dtype='complex64')
    for i in range(N):
        for j in range(K):
            ex = 2 * (i + 1) * (j + 1) * math.pi / K
            inverse_D[i][j] = complex(math.cos(ex), math.sin(ex))
    X = np.matmul(inverse_D, cX)
    X = X / K
    X = np.abs(X)

    # generalized inverse matrix of C
    inv_C = np.linalg.inv(C)
    X = np.matmul(inv_C, X)

    # Reverse the framing process.
    x = np.zeros(shape=(N + (N_frames - 1) * (N - int(0.01 * sr)), ))
    x[0: N] = X[:, 0]
    indexes = np.array([N - int(0.01 * sr), N - 1, 2 * N - int(0.01 * sr) - 1])
    for i in range(N_frames - 1):
        x[indexes[0]: indexes[2] + 1] += X[:, i + 1]
        x[indexes[0]: indexes[1] + 1] /= 2
        indexes = indexes + N - int(0.01 * sr)
        if indexes[2] >= len(x):
            indexes[2] = len(x) - 1
    x = x[0: len(song)]

    # Reverse the pre-emphasis process.
    for i in range(len(x) - 1):
        x[i + 1] += 0.97 * x[i]
    
    # normalization
    x_range = np.max(x) - np.min(x)
    x_mean = np.mean(x)
    for i in range(len(x)):
        x[i] = (x[i] - x_mean) / x_range
    
    write_wav(path=output_path, y=x, sr=sr)
    convert(path=output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="An attack method based on orthogonal projection")
    parser.add_argument("--song_index", type=int, default=1, help="choose a song based on its index")
    parser.add_argument("--command_index", type=int, default=1, help="choose a command based on its index")
    parser.add_argument("--origin", type=float, default=1.0, help="origin of the song to inject the command")
    arguments = parser.parse_args()
    main(arguments)
