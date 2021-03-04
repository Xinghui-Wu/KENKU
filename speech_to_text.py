import argparse
import os
import csv

import speech_recognition as sr
from aip import AipSpeech

from aliyun import aliyun_asr
from tencent import tencent_asr
from iflytek import RequestApi


BAIDU = AipSpeech(appId="23655665", 
                  apiKey="GoTmvRNcwjt6aPjZg4v7h26h", 
                  secretKey="G1q0V4qBGMCAKD3yDCPrb9tWj6ooW2qn")


def speech_to_text(directory, transcriptions, proxy, language, output):
    """Use the text-to-speech APIs of various manufactures to recognize audio files.
    Call various APIs to recognize the audio files in the specified directory.
    Output the recognition results of various APIs with the target transcriptions in a csv file.

    Args:
        directory (str): Input directory of the audio files to be recognized.
        transcriptions (str): Path of the transcription file of the corresponding audio files.
        proxy (int): Set 1 to connect APIs with proxy.
        language (str): Chinese or English.
        output (str): Output path of the recognition result file in csv format.
    """
    if proxy == 0:
        asr_results = [("filename", "transcription", "Baidu", "Aliyun", "Tencent", "iFLYTEK")]
    else:
        asr_results = [("filename", "transcription", "Google", "CMU Sphinx")]
        output = output[: -4] + "-Proxy.csv"

    audio_filenames = os.listdir(directory)
    audio_filenames.sort()

    with open(transcriptions, 'r') as transcriptions_txt:
        transcriptions = transcriptions_txt.readlines()
    
    for i in range(len(audio_filenames)):
        print("****************************************************************************************************")
        print(audio_filenames[i])
        print()

        audio_path = os.path.join(directory, audio_filenames[i])
        
        if proxy == 0:
            asr_results.append((audio_filenames[i], 
                                transcriptions[i].strip('\n'), 
                                baidu_speech_to_text(audio_path, language), 
                                aliyun_speech_to_text(audio_path, language), 
                                tencent_speech_to_text(audio_path, language), 
                                iflytek_speech_to_text(audio_path, language)))
        else:
            asr_results.append((audio_filenames[i], 
                                transcriptions[i].strip('\n'), 
                                google_speech_to_text(audio_path, language), 
                                cmu_sphinx_speech_to_text(audio_path, language)))
        
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
        app_key = "yH4fwRLGRZP7CKfz"
    else:
        app_key = "LdxxwQtP3MyRZYqU"
    
    asr_result = aliyun_asr(app_key=app_key, 
                            access_key_id="LTAI4GEGDb8Tqyr6XrPhU6ac", 
                            access_key_secret="F2BdzQjqKgNUkJHrsP2Z1W0ITKIpCr",
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
    asr_result = tencent_asr(secret_id="AKIDVs2ze24Afy1ojor8MVNyHYaLp3IItTdK", 
                             secret_key="buyf8KmxmKbbKcdOsugBPKhtoagWrviG",
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
    iFlytek_asr = RequestApi(appid="604043d3", 
                             secret_key="a775c26b59359599328dfb790b58fa04", 
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


def google_speech_to_text(audio_path, language):
    """Use the text-to-speech API of Google to recognize an audio file.

    Args:
        audio_path (str): Path of an audio file to be recognized.
        language (str): Chinese or English.

    Returns:
        str: Recognition result of Google's API.
    """
    r = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = r.record(source)
    
    if language == "Chinese":
        language = "zh-CN"
    else:
        language = "en-US"
    
    asr_result = ""
    
    try:
        print("Google Speech Recognition thinks you said " + r.recognize_google(audio_data=audio, language=language))
        asr_result = r.recognize_google(audio_data=audio, language=language)
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))
    
    return asr_result


def cmu_sphinx_speech_to_text(audio_path, language):
    """Use the text-to-speech API of CMU Sphinx to recognize an audio file.

    Args:
        audio_path (str): Path of an audio file to be recognized.
        language (str): Chinese or English.

    Returns:
        str: Recognition result of CMU Sphinx's API.
    """
    r = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = r.record(source)
    
    if language == "Chinese":
        language = "zh-CN"
    else:
        language = "en-US"
    
    asr_result = ""
    
    try:
        print("Sphinx thinks you said " + r.recognize_sphinx(audio_data=audio, language=language))
        asr_result = r.recognize_sphinx(audio_data=audio, language=language)
    except sr.UnknownValueError:
        print("Sphinx could not understand audio")
    except sr.RequestError as e:
        print("Sphinx error; {0}".format(e))
    
    return asr_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Use the text-to-speech APIs of various manufactures to recognize audio files.")
    parser.add_argument("-d", "--directory", type=str, default="./Audio Samples/Malicious Samples/", help="Input directory of the audio files to be recognized.")
    parser.add_argument("-t", "--transcriptions", type=str, default="./Audio Samples/Malicious-Commands.txt", help="Path of the transcription file of the corresponding audio files.")
    parser.add_argument("-p", "--proxy", type=int, default=0, help="Set 1 to connect APIs with proxy.")
    parser.add_argument("-l", "--language", type=str, default="English", help="Chinese or English.")
    parser.add_argument("-o", "--output", type=str, default="./Audio Samples/ASR-Results.csv", help="Output path of the recognition result file in csv format.")
    args = parser.parse_args()

    speech_to_text(directory=args.directory, transcriptions=args.transcriptions, 
                   proxy=args.proxy, language=args.language, output=args.output)
