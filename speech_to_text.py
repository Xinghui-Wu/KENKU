import argparse
import os
import csv

from aip import AipSpeech

from alibaba import alibaba_asr
from tencent import tencent_asr
from iflytek import RequestApi


# Set the Baidu AI client with the account information.
BAIDU = AipSpeech(appId="", 
                  apiKey="", 
                  secretKey="")


def speech_to_text(directory, transcriptions, language, output):
    """Use the text-to-speech APIs of various manufactures to recognize audio files.
    Call various APIs to recognize the audio files in the specified directory.
    Output the recognition results of various APIs with the target transcriptions in a csv file.

    Args:
        directory (str): Input directory of the audio files to be recognized.
        transcriptions (str): Path of the transcription file of the corresponding audio files.
        language (str): Chinese or English.
        output (str): Output path of the recognition result file in csv format.
    """
    asr_results = [("filename", "transcription", "Baidu", "Aliyun", "Tencent", "iFLYTEK")]

    audio_filenames = os.listdir(directory)
    audio_filenames.sort()

    with open(transcriptions, 'r') as transcriptions_txt:
        transcriptions = transcriptions_txt.readlines()
    
    for i in range(len(audio_filenames)):
        print("****************************************************************************************************")
        print(audio_filenames[i])
        print()

        audio_path = os.path.join(directory, audio_filenames[i])
        
        asr_results.append((audio_filenames[i], 
                            transcriptions[i].strip('\n'), 
                            baidu_speech_to_text(audio_path, language), 
                            aliyun_speech_to_text(audio_path, language), 
                            tencent_speech_to_text(audio_path, language),
                            iflytek_speech_to_text(audio_path, language)))
        
        print("****************************************************************************************************")
        print()

    with open(output, 'w') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerows(asr_results)


def baidu_speech_to_text(audio_path, language):
    """Use the text-to-speech API of Baidu to recognize an audio file.

    Args:
        audio_path (str): Path of an audio file to be recognized.
        language (str): Chinese or English.

    Returns:
        str: Recognition result of Baidu's API.
    """
    if language == "Chinese":
        language = 1537
    else:
        language = 1737
    
    with open(audio_path, 'rb') as audio_file:
        audio = audio_file.read()
    
    asr_result = BAIDU.asr(audio, 'wav', 16000, {'dev_pid': language})
    print(asr_result)

    if 'result' in asr_result:
        return asr_result['result'][0]
    else:
        return ""


def aliyun_speech_to_text(audio_path, language):
    """Use the text-to-speech API of Aliyun to recognize an audio file.

    Args:
        audio_path (str): Path of an audio file to be recognized.
        language (str): Chinese or English.

    Returns:
        str: Recognition result of Aliyun's API.
    """
    if language == "Chinese":
        app_key = ""
    else:
        app_key = ""
    
    asr_result = alibaba_asr(app_key=app_key, 
                             access_key_id="", 
                             access_key_secret="",
                             audio_path=audio_path)
    
    return asr_result


def tencent_speech_to_text(audio_path, language):
    """Use the text-to-speech API of Tencent to recognize an audio file.

    Args:
        audio_path (str): Path of an audio file to be recognized.
        language (str): Chinese or English.

    Returns:
        str: Recognition result of Tencent's API.
    """
    asr_result = tencent_asr(secret_id="", 
                             secret_key="",
                             audio_path=audio_path, language=language)
    
    return asr_result


def iflytek_speech_to_text(audio_path, language):
    """Use the text-to-speech API of iFLYTEK to recognize an audio file.

    Args:
        audio_path (str): Path of an audio file to be recognized.
        language (str): Chinese or English.

    Returns:
        str: Recognition result of iFLYTEK's API.
    """
    iFlytek_asr = RequestApi(appid="", 
                             secret_key="", 
                             upload_file_path=audio_path, language=language)

    asr_result = iFlytek_asr.all_api_request()

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
    parser = argparse.ArgumentParser(description="Use the text-to-speech APIs of various manufactures to recognize audio files.")
    parser.add_argument("-d", "--directory", type=str, default="./Audio Samples/Malicious Samples/", help="Input directory of the audio files to be recognized.")
    parser.add_argument("-t", "--transcriptions", type=str, default="./Audio Samples/Malicious-Commands.txt", help="Path of the transcription file of the corresponding audio files.")
    parser.add_argument("-l", "--language", type=str, default="English", help="Chinese or English.")
    parser.add_argument("-o", "--output", type=str, default="./Audio Samples/ASR-Results.csv", help="Output path of the recognition result file in csv format.")
    
    args = parser.parse_args()

    speech_to_text(directory=args.directory, transcriptions=args.transcriptions, language=args.language, output=args.output)
