import argparse
import os
import random

from aip import AipSpeech
from pydub import AudioSegment


APP_ID = "17156719"
API_KEY = "aqZ67xX12E6umtTXfw44kYi4"
SECRET_KEY = "w2fLg5VzQMcumFHgtPpvsXsPAj65FUyw"

CLIENT = AipSpeech(APP_ID, API_KEY, SECRET_KEY)


def generate_commands(commands, directory):
    """[summary]

    Args:
        commands ([type]): [description]
        directory ([type]): [description]
    """
    with open(commands, 'r') as commands_txt:
        commands = commands_txt.readlines()
    
    for i in range(len(commands)):
        text = commands[i].strip('\n')
        mp3_path = os.path.join(directory, "Command-{}.mp3".format(i))
        wav_path = os.path.join(directory, "Command-{}.wav".format(i))

        text_to_speech(text, mp3_path, wav_path)


def text_to_speech(text, mp3_path, wav_path):
    """[summary]

    Args:
        text ([type]): [description]
        mp3_path ([type]): [description]
        wav_path ([type]): [description]
    """
    command = CLIENT.synthesis(text, 'zh', 1, {'vol': 5, 'per': random.choice([0, 1])})
    
    if not isinstance(command, dict):
        with open(mp3_path, 'wb') as mp3_file:
            mp3_file.write(command)
        
        command = AudioSegment.from_file(file=mp3_path, format='mp3')
        command.export(out_f=wav_path, format='wav')
        os.remove(mp3_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-c", "--commands", type=str, default="./Audio Samples/Commands.txt", help="")
    parser.add_argument("-d", "--directory", type=str, default="./Audio Samples/Commands/", help="")
    args = parser.parse_args()

    generate_commands(commands=args.commands, directory=args.directory)
