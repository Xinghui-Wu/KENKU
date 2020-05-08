import math
import random
import numpy as np
import scipy
from librosa import load
from librosa.output import write_wav
from scipy.io.wavfile import write

from algebraic_mfcc import algebraic_mfcc


def main():
    song_path = ROOT_DIRECTORY + "Song.wav"
    command_path = ROOT_DIRECTORY + "Command.wav"
    output_path = ROOT_DIRECTORY + "Attack.wav"
    attack(song_path, command_path, output_path)


def attack(song_path, command_path, output_path):
    sr = 16000
    song, _ = load(song_path, sr=sr)
    command, _ = load(command_path, sr=sr)
    
    # 目标指令嵌入原点
    origin = 1
    # 截取与目标指令等长的歌曲片段
    song = song[int(origin * sr): int(origin * sr) + len(command)]

    # 提取MFCC和各常矩阵
    A, C, D, E, F, F1, song_MFCC = algebraic_mfcc(x=song, sr=sr)
    A, C, D, E, F, F1, command_MFCC = algebraic_mfcc(x=command, sr=sr)

    # 根据矩阵乘法结合律，令G=F1*F
    G = np.matmul(F1, F)
    # 求矩阵G列向量子空间的一组标准正交基
    orth_G = scipy.linalg.orth(G)
    # 将矩阵command_MFCC的各列向矩阵G的列向量子空间正交投影
    projection = np.matmul(orth_G.T, command_MFCC)

    # 矩阵G的广义逆矩阵
    pinv_G = np.linalg.pinv(G)
    X = np.matmul(pinv_G, projection)
    X = np.exp(X)

    # 矩阵E的广义逆矩阵
    pinv_E = np.linalg.pinv(E)
    X = np.matmul(pinv_E, X)

    # 随机拆解平方和
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
    
    # 离散傅里叶变换的逆变换矩阵
    N = D.shape[1]
    inverse_D = np.zeros(shape=(N, K), dtype='complex64')
    for i in range(N):
        for j in range(K):
            ex = 2 * (i + 1) * (j + 1) * math.pi / K
            inverse_D[i][j] = complex(math.cos(ex), math.sin(ex))
    X = np.matmul(inverse_D, cX)
    X = X / K
    # 
    X = np.abs(X)

    # 矩阵C的逆矩阵
    inv_C = np.linalg.inv(C)
    X = np.matmul(inv_C, X)

    # 逆向分帧结果
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

    # 逆向预加重
    for i in range(len(x) - 1):
        x[i + 1] += 0.97 * x[i]
    
    # 归一化
    x_range = np.max(x) - np.min(x)
    x_mean = np.mean(x)
    for i in range(len(x)):
        x[i] = (x[i] - x_mean) / x_range
    
    write_wav(path=output_path, y=x, sr=sr)
    convert(path=output_path)


def convert(path):
    audio, _ = load(path=path, sr=16000)
    audio *= 32767
    audio = audio.astype(np.int16)
    write(filename=path, rate=16000, data=audio)


if __name__ == '__main__':
    ROOT_DIRECTORY = "./Audio Samples/"
    main()
