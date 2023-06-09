import argparse
import json
import os
import random
import logging

from gtts import gTTS
from aip import AipSpeech
from pydub import AudioSegment

from utils import *

logging.basicConfig(level=logging.INFO, format="%(asctime)s \t %(message)s", datefmt="%Y-%m-%d %H:%M:%S")


def text_to_speech(commands, dir, language):
    command_table = [["audio path", "transcription"]]
    
    with open(commands, 'r') as commands_txt:
        commands = commands_txt.readlines()
    
    if not os.path.exists(dir):
        os.makedirs(dir)
    
    length = len(str(len(commands)))

    for i in range(len(commands)):
        text = commands[i].strip('\n')

        filename = str(i + 1).zfill(length)
        mp3_path = os.path.abspath(os.path.join(dir, "{}.mp3".format(filename)))
        wav_path = os.path.abspath(os.path.join(dir, "{}.wav".format(filename)))

        logging.info("Synthesize {}: {}".format(filename, text))

        if language == 0:
            google_text_to_speech(text, mp3_path)
        else:
            baidu_text_to_speech(text, mp3_path)
        
        # Convert the mp3 file to the wav format.
        command = AudioSegment.from_file(file=mp3_path, format='mp3')
        command = command.set_channels(1).set_frame_rate(16000)
        command.export(out_f=wav_path, format='wav')
        
        os.remove(mp3_path)

        command_table.append([wav_path, text])
    
    write_csv(csv_path=os.path.join(dir, "commands.csv"), lines=command_table)


def google_text_to_speech(text, mp3_path):
    command = gTTS(text=text, lang="en")
    command.save(mp3_path)


def baidu_text_to_speech(text, mp3_path):
    baidu = AipSpeech(ACCOUNT["Baidu"]["appId"], ACCOUNT["Baidu"]["apiKey"], ACCOUNT["Baidu"]["secretKey"])

    command = baidu.synthesis(text, 'zh', 1, {'vol': 5, 'per': random.choice([0, 1])})

    if not isinstance(command, dict):
        with open(mp3_path, 'wb') as mp3_file:
            mp3_file.write(command)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Use the text-to-speech APIs of Baidu and Google to synthesize audio files.")
    parser.add_argument("-c", "--commands", type=str, default="commands/commands.txt", help="Path of the transcription file of the desired commands.")
    parser.add_argument("-d", "--dir", type=str, default="commands/", help="Output directory of the command audio files.")
    parser.add_argument("-l", "--language", type=int, default=0, help="Specify 0 or 1 to synthesize audio files for English or Chinese commands respectively.")
    parser.add_argument("-a", "--account_json", type=str, default="account.json", help="Path of the account information file.")

    args = parser.parse_args()

    # Read the account info in the json file.
    with open(args.account_json, 'r') as account_json:
        ACCOUNT = json.load(account_json)

    text_to_speech(commands=args.commands, dir=args.dir, language=args.language)
