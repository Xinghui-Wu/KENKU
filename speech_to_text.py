import argparse
import os
import csv

from aip import AipSpeech


APP_ID = "17156719"
API_KEY = "aqZ67xX12E6umtTXfw44kYi4"
SECRET_KEY = "w2fLg5VzQMcumFHgtPpvsXsPAj65FUyw"

CLIENT = AipSpeech(APP_ID, API_KEY, SECRET_KEY)


def speech_to_text(directory, transcriptions, output):
    """[summary]

    Args:
        directory ([type]): [description]
        transcriptions ([type]): [description]
        output ([type]): [description]
    """
    asr_results = [("filename", "transcription", "Baidu", "iFLYTEK")]

    audio_filenames = os.listdir(directory)
    audio_filenames.sort()

    with open(transcriptions, 'r') as transcription_txt:
        transcriptions = transcription_txt.readlines()
    
    for i in range(len(audio_filenames)):
        audio_path = os.path.join(directory, audio_filenames[i])

        asr_results.append((audio_filenames[i], 
                            transcriptions[i].strip('\n'), 
                            baidu_speech_to_text(audio_path), 
                            iflytek_speech_to_text(audio_path)))

    with open(output, 'w') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerows(asr_results)


def baidu_speech_to_text(audio_path):
    """[summary]

    Args:
        audio_path ([type]): [description]

    Returns:
        [type]: [description]
    """
    with open(audio_path, 'rb') as audio_file:
        audio = audio_file.read()
    
    asr_result = CLIENT.asr(audio, 'wav', 16000, {'dev_pid': 1537})

    if 'result' in asr_result:
        return asr_result['result'][0]
    else:
        return ""


def iflytek_speech_to_text(audio_path):
    """[summary]

    Args:
        audio_path ([type]): [description]

    Returns:
        [type]: [description]
    """
    return ""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-d", "--directory", type=str, default="./Audio Samples/Malicious Samples/", help="")
    parser.add_argument("-t", "--transcriptions", type=str, default="./Audio Samples/Commands.txt", help="")
    parser.add_argument("-o", "--output", type=str, default="./Audio Samples/ASR_Results.csv", help="")
    args = parser.parse_args()

    speech_to_text(directory=args.directory, transcriptions=args.transcriptions, output=args.output)
