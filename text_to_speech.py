import argparse
import os
import random

from aip import AipSpeech
from gtts import gTTS
from pydub import AudioSegment


BAIDU = AipSpeech(appId="17156719", 
                  apiKey="aqZ67xX12E6umtTXfw44kYi4", 
                  secretKey="w2fLg5VzQMcumFHgtPpvsXsPAj65FUyw")


def generate_commands(commands, directory, language):
    """Synthesize the given command texts into command audio files.

    Args:
        commands (str): Path of the transcription file of the desired commands.
        directory (str): Output directory of some command files.
        language (str): Chinese or English.
    """
    with open(commands, 'r') as commands_txt:
        commands = commands_txt.readlines()
    
    for i in range(len(commands)):
        text = commands[i].strip('\n')
        mp3_path = os.path.join(directory, "Command-{}.mp3".format(i))
        wav_path = os.path.join(directory, "Command-{}.wav".format(i))

        if language == "Chinese":
            baidu_text_to_speech(text, mp3_path, wav_path)
        else:
            google_text_to_speech(text, mp3_path)


def baidu_text_to_speech(text, mp3_path, wav_path):
    """Use the text-to-speech API of Baidu to synthesize the given Chinese command text into a command audio file.

    Args:
        text (str): The given command text.
        mp3_path (str): Output path of the mp3 file.
        wav_path (str): Output path of the converted wav file.
    """
    command = BAIDU.synthesis(text, 'zh', 1, {'vol': 5, 'per': random.choice([0, 1])})
    
    if not isinstance(command, dict):
        with open(mp3_path, 'wb') as mp3_file:
            mp3_file.write(command)
        
        command = AudioSegment.from_file(file=mp3_path, format='mp3')
        command.export(out_f=wav_path, format='wav')
        os.remove(mp3_path)


def google_text_to_speech(text, mp3_path):
    """Use the text-to-speech API of Google to synthesize the given English command text into a command audio file.

    Args:
        text (str): The given command text.
        mp3_path (str): Output path of the mp3 file.
    """
    command = gTTS(text=text, lang="en")
    command.save(mp3_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Use the text-to-speech API of Baidu to synthesize audio files.")
    parser.add_argument("-c", "--commands", type=str, default="./Audio Samples/Commands.txt", help="Path of the transcription file of the desired commands.")
    parser.add_argument("-d", "--directory", type=str, default="./Audio Samples/Commands/", help="Output directory of some command files.")
    parser.add_argument("-l", "--language", type=str, default="Chinese", help="Chinese or English.")
    args = parser.parse_args()

    generate_commands(commands=args.commands, directory=args.directory, language=args.language)
