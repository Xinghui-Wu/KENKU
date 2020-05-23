import os
import random
from aip import AipSpeech
from pydub import AudioSegment


def main():
    APP_ID = "17156719"
    API_KEY = "aqZ67xX12E6umtTXfw44kYi4"
    SECRET_KEY = "w2fLg5VzQMcumFHgtPpvsXsPAj65FUyw"

    client = AipSpeech(APP_ID, API_KEY, SECRET_KEY)

    command_list = [["给小明转账一万元", "./Audio Samples/Commands/Command-1.wav", "./Audio Samples/Commands/Command-1.mp3"],
                    ["清除手机数据", "./Audio Samples/Commands/Command-2.wav", "./Audio Samples/Commands/Command-2.mp3"],
                    ["拨打110", "./Audio Samples/Commands/Command-3.wav", "./Audio Samples/Commands/Command-3.mp3"],
                    ["手机关机", "./Audio Samples/Commands/Command-4.wav", "./Audio Samples/Commands/Command-4.mp3"],
                    ["关闭无线网络", "./Audio Samples/Commands/Command-5.wav", "./Audio Samples/Commands/Command-5.mp3"]]

    for i in range(len(command_list)):
        generate_command(client, command_list[i][0], command_list[i][1], command_list[i][2])


def generate_command(client, command_text, wav_command_path, mp3_command_path):
    command = client.synthesis(command_text, 'zh', 1, {'vol': 5, 'per': random.choice([0, 1])})
    if not isinstance(command, dict):
        with open(mp3_command_path, 'wb') as f:
            f.write(command)
        command = AudioSegment.from_file(file=mp3_command_path, format='mp3')
        command.export(out_f=wav_command_path, format='wav')
        os.remove(mp3_command_path)


if __name__ == '__main__':
    main()
