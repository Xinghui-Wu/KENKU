import math
import numpy as np
from librosa import load
from librosa.filters import mel


def main():
    n = 16000
    x, sr = load(path="./Audio Samples/Command.wav", sr=n)
    x.shape = (len(x), 1)
    A, x = pre_emphasis(x)
    X = framing(x, sr=n)
    print(X.shape)
    C, X = windowing(X)
    print(X.shape)
    D, X = DFT(X)
    print(X.shape)
    E, X = log_energy(X=X, sr=n)
    print(X.shape)
    F, F1, MFCC = DCT(X=X)
    print(MFCC.shape)
    print(MFCC)


def pre_emphasis(x, mu=-0.97):
    n = len(x)

    A = np.eye(n)
    A[0][0] = 0
    for i in range(n - 1):
        A[i + 1][i] = mu

    return A, np.matmul(A, x)


def framing(x, sr, frame_duration=0.025, frame_overlap=0.01):
    n = len(x)
    N = int(frame_duration * sr)
    alpha = int(frame_overlap * sr)
    N_frames = math.ceil((n - N) / (N - alpha) + 1)
    n1 = N + (N_frames - 1) * (N - alpha)

    if n1 > n:
        zero_padding = np.zeros(shape=(n1 - n, 1))
        x = np.vstack((x, zero_padding))

    X = np.zeros(shape=(N, N_frames))
    p = 0
    for frame in range(N_frames):
        for i in range(N):
            X[i][frame] = x[p]
            p = p + 1
        p = p - alpha

    return X


def windowing(X, omega=0.46):
    N = len(X)

    C = np.zeros(shape=(N, N))
    for i in range(N):
        # C[i][i] = 1 - omega - omega * math.cos(2 * (i + 1) * math.pi / (N - 1))
        C[i][i] = 1 - omega - omega * math.cos(2 * i * math.pi / (N - 1))

    return C, np.matmul(C, X)


def DFT(X, K=512):
    N = len(X)

    D = np.zeros(shape=(K, N), dtype='complex64')
    for i in range(K):
        for j in range(N):
            ex = -2 * (i + 1) * (j + 1) * math.pi / N
            D[i][j] = complex(math.cos(ex), math.sin(ex))

    return D, np.matmul(D, X)


def log_energy(X, sr, M=26):
    K, N_frames = X.shape

    E = mel(sr=sr, n_fft=2*(K-1), n_mels=M)

    X = np.square(np.abs(X)) / K
    for i in range(K):
        for j in range(N_frames):
            if X[i][j] == 0:
                X[i][j] = 1e-200

    return E, np.log(np.matmul(E, X))


def DCT(X, n_mfccs=12, begin=1):
    M = len(X)

    F = np.zeros(shape=(M, M))
    for i in range(M):
        for j in range(M):
            F[i][j] = math.cos(math.pi / M * i * (j + 0.5))

    F1 = np.zeros(shape=(n_mfccs, M))
    for i in range(n_mfccs):
        F1[i][begin + i] = 1

    return F, F1, np.matmul(F1, np.matmul(F, X))


if __name__ == '__main__':
    main()
