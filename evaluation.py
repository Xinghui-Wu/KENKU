import matplotlib.pyplot as plt
from aip import AipSpeech


def main():
    client = AipSpeech(APP_ID, API_KEY, SECRET_KEY)
    # recognize_speech(client=client, audio_path=ROOT_DIRECTORY+"Song-44.1kHz.wav")
    recognize_speech(client=client, audio_path=ROOT_DIRECTORY+"Command.wav")


def recognize_speech(client, audio_path):
    with open(audio_path, 'rb') as fp:
        audio = fp.read()
    result = client.asr(audio, 'wav', 16000, {'dev_pid': 1537})
    print(result)
    print()
    return result


def plot_mfcc(song_mfcc, command_mfcc):
    interval_radius = 25
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.imshow(song_mfcc, vmin=-interval_radius, vmax=interval_radius)
    plt.colorbar()
    plt.title("Song MFCC")
    plt.subplot(2, 1, 2)
    plt.imshow(command_mfcc, vmin=-interval_radius, vmax=interval_radius)
    plt.colorbar()
    plt.title("Command MFCC")
    plt.show()


if __name__ == '__main__':
    ROOT_DIRECTORY = "./Audio Samples/"

    APP_ID = "17156719"
    API_KEY = "aqZ67xX12E6umtTXfw44kYi4"
    SECRET_KEY = "w2fLg5VzQMcumFHgtPpvsXsPAj65FUyw"

    main()
