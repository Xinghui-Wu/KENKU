import argparse
import os
import csv

from aip import AipSpeech

from iflytek import RequestApi


BAIDU = AipSpeech(appId="17156719", 
                  apiKey="aqZ67xX12E6umtTXfw44kYi4", 
                  secretKey="w2fLg5VzQMcumFHgtPpvsXsPAj65FUyw")

IFLYTEK_APPID = "5fb0c096"
IFLYTEK_SECRET_KEY = "4e5779d189ece9a31a0d526054513cc2"


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

    with open(transcriptions, 'r') as transcriptions_txt:
        transcriptions = transcriptions_txt.readlines()
    
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
    
    asr_result = BAIDU.asr(audio, 'wav', 16000, {'dev_pid': 1537})

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
    iFlytek_asr = RequestApi(appid=IFLYTEK_APPID, secret_key=IFLYTEK_SECRET_KEY, upload_file_path=audio_path)

    asr_result = iFlytek_asr.all_api_request()

    if 'data' in asr_result:
        asr_result = asr_result['data']

        if asr_result != None:
            return eval(asr_result)[0]['onebest']

    return ""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-d", "--directory", type=str, default="./Audio Samples/Malicious Samples/", help="")
    parser.add_argument("-t", "--transcriptions", type=str, default="./Audio Samples/Malicious-Commands.txt", help="")
    parser.add_argument("-o", "--output", type=str, default="./Audio Samples/ASR-Results.csv", help="")
    args = parser.parse_args()

    speech_to_text(directory=args.directory, transcriptions=args.transcriptions, output=args.output)
