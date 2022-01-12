import argparse
import json
import io
import logging
import traceback

import azure.cognitiveservices.speech as speechsdk
from aip import AipSpeech
from google.cloud import speech

from utils import *
from asr.alibaba import alibaba
from asr.tencent import tencent
from asr.iflytek import iFLYTEK

logging.basicConfig(level=logging.INFO, format="%(asctime)s \t %(message)s", datefmt="%Y-%m-%d %H:%M:%S")


def black_box_asr(input_csv, output_csv, language, vendor):
    assert language == 0 or language == 1
    assert vendor == 0 or vendor == 1
    
    audio_table = read_csv(csv_path=input_csv)

    with open(output_csv, 'w') as csv_file:
        csv_writer = csv.writer(csv_file)

        title = [["Google", "Microsoft"], ["Alibaba", "Tencent", "iFLYTEK"]]
        audio_table[0].extend(title[vendor])
        csv_writer.writerow(audio_table[0])
        
        for audio_info in audio_table[1: ]:
            audio_path = audio_info[0]
            
            logging.info("Start to transcribe {}".format(audio_path))

            try:
                if vendor == 0:
                    asr_results = [google_asr(audio_path, language)]
                else:
                    asr_results = [alibaba_asr(audio_path, language), tencent_asr(audio_path, language), iflytek_asr(audio_path, language)]
            
                audio_info.extend(asr_results)
            except:
                logging.error("Error in {}".format(audio_path))
                traceback.print_exc()
            finally:
                csv_writer.writerow(audio_info)


def baidu_asr(audio_path, language):
    baidu = AipSpeech(ACCOUNT["Baidu"]["appId"], ACCOUNT["Baidu"]["apiKey"], ACCOUNT["Baidu"]["secretKey"])
    
    with open(audio_path, 'rb') as audio_file:
        audio = audio_file.read()
    
    if language == 0:
        language = 1737
    else:
        language = 1537
    
    asr_result = baidu.asr(audio, 'wav', 16000, {'dev_pid': language})
    print(asr_result)

    if 'result' in asr_result:
        return asr_result['result'][0]
    else:
        return ""


def alibaba_asr(audio_path, language):
    if language == 0:
        app_key = ACCOUNT["Alibaba"]["app_key_en"]
    else:
        app_key = ACCOUNT["Alibaba"]["app_key_zh"]
    
    asr_result = alibaba(app_key, ACCOUNT["Alibaba"]["access_key_id"], ACCOUNT["Alibaba"]["access_key_secret"], audio_path)
    
    return asr_result


def tencent_asr(audio_path, language):
    asr_result = tencent(ACCOUNT["Tencent"]["secret_id"], ACCOUNT["Tencent"]["secret_key"], audio_path, language)
    
    return asr_result


def iflytek_asr(audio_path, language):
    iflytek = iFLYTEK(ACCOUNT["iFLYTEK"]["appid"], ACCOUNT["iFLYTEK"]["secret_key"], audio_path, language)

    asr_result = iflytek.all_api_request()

    if 'data' in asr_result:
        asr_result = asr_result['data']

        if asr_result != None:
            try:
                asr_result = eval(asr_result)[0]['onebest']
            except:
                asr_result = ""
            finally:
                return asr_result

    return ""


def google_asr(audio_path, language):
    client = speech.SpeechClient()

    with io.open(audio_path, "rb") as audio_file:
        content = audio_file.read()
    
    audio = speech.RecognitionAudio(content=content)

    if language == 0:
        language_code = "en-US"
    else:
        language_code = "zh"
    
    config = speech.RecognitionConfig(
        sample_rate_hertz=16000,
        language_code=language_code,
    )

    response = client.recognize(config=config, audio=audio)

    # Each result is for a consecutive portion of the audio. Iterate through
    # them to get the transcripts for the entire audio file.
    
    for result in response.results:
        # The first alternative is the most likely one for this portion.
        print(u"Transcript: {}".format(result.alternatives[0].transcript))

        return result.alternatives[0].transcript


def microsoft_asr(audio_path, language):
    if language == 0:
        speech_recognition_language = "en-US"
    else:
        speech_recognition_language = "zh-CN"
    
    speech_config = speechsdk.SpeechConfig(subscription=ACCOUNT["Microsoft"]["speech_key"], region=ACCOUNT["Microsoft"]["location"], speech_recognition_language=speech_recognition_language)
    audio_input = speechsdk.AudioConfig(filename=audio_path)
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_input)
    
    result = speech_recognizer.recognize_once_async().get()

    asr_result = ""
    
    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        asr_result = result.text
        print("Recognized: {}".format(result.text))
    elif result.reason == speechsdk.ResultReason.NoMatch:
        print("No speech could be recognized: {}".format(result.no_match_details))
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        print("Speech Recognition canceled: {}".format(cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            print("Error details: {}".format(cancellation_details.error_details))

    return asr_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-i", "--input_csv", type=str, default="input.csv", help="")
    parser.add_argument("-o", "--output_csv", type=str, default="output.csv", help="")
    parser.add_argument("-l", "--language", type=int, default=0, help="")
    parser.add_argument("-v", "--vendor", type=int, default=1, help="")
    parser.add_argument("-a", "--account_json", type=str, default=".vscode/account.json", help="")

    args = parser.parse_args()

    # Read the account info in the json file.
    with open(args.account_json, 'r') as account_json:
        ACCOUNT = json.load(account_json)
    
    black_box_asr(input_csv=args.input_csv, output_csv=args.output_csv, language=args.language, vendor=args.vendor)
