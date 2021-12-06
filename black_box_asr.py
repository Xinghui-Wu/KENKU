import argparse
import json
import logging
import traceback

from aip import AipSpeech

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

        title = [["Google", "Microsoft", "Amazon", "IBM"], ["Baidu", "Alibaba", "Tencent", "iFLYTEK"]]
        audio_table[0].extend(title[vendor])
        csv_writer.writerow(audio_table[0])
        
        for audio_info in audio_table[1: ]:
            audio_path = audio_info[0]
            
            logging.info("Start to transcribe {}".format(audio_path))

            try:
                if vendor == 0:
                    asr_results = ["Google", "Microsoft", "Amazon", "IBM"]
                else:
                    asr_results = [baidu_asr(audio_path, language), alibaba_asr(audio_path, language), tencent_asr(audio_path, language), iflytek_asr(audio_path, language)]
            
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
