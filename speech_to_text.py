import argparse
import os
import csv

from aip import AipSpeech

from iflytek import RequestApi


BAIDU = AipSpeech(appId="17156719", 
                  apiKey="aqZ67xX12E6umtTXfw44kYi4", 
                  secretKey="w2fLg5VzQMcumFHgtPpvsXsPAj65FUyw")


def speech_to_text(directory, transcriptions, output):
    """Use the text-to-speech APIs of various manufactures to recognize audio files.
    Call various APIs to recognize the audio files in the specified directory.
    Output the recognition results of various APIs with the target transcriptions in a csv file.

    Args:
        directory (str): Input directory of the audio files to be recognized.
        transcriptions (str): Path of the transcription file of the corresponding audio files.
        output (str): Output path of the recognition result file in csv format.
    """
    asr_results = [("filename", "transcription", "Baidu", "iFLYTEK")]

    audio_filenames = os.listdir(directory)
    audio_filenames.sort(key=lambda filename: os.path.getmtime(os.path.join(directory, filename)))

    with open(transcriptions, 'r') as transcriptions_txt:
        transcriptions = transcriptions_txt.readlines()
    
    for i in range(len(audio_filenames)):
        print("****************************************************************************************************")
        print(audio_filenames[i])
        print()

        audio_path = os.path.join(directory, audio_filenames[i])

        asr_results.append((audio_filenames[i], 
                            transcriptions[i].strip('\n'), 
                            baidu_speech_to_text(audio_path), 
                            iflytek_speech_to_text(audio_path)))
        
        print("****************************************************************************************************")
        print()

    with open(output, 'w') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerows(asr_results)


def baidu_speech_to_text(audio_path):
    """Use the text-to-speech API of Baidu to recognize an audio file.

    Args:
        audio_path (str): Path of an audio file to be recognized.

    Returns:
        str: Recognition result of Baidu's API.
    """
    with open(audio_path, 'rb') as audio_file:
        audio = audio_file.read()
    
    asr_result = BAIDU.asr(audio, 'wav', 16000, {'dev_pid': 1537})

    if 'result' in asr_result:
        return asr_result['result'][0]
    else:
        return ""


def iflytek_speech_to_text(audio_path):
    """Use the text-to-speech API of iFLYTEK to recognize an audio file.

    Args:
        audio_path (str): Path of an audio file to be recognized.

    Returns:
        str: Recognition result of iFLYTEK's API.
    """
    iFlytek_asr = RequestApi(appid="5fb0c096", 
                             secret_key="4e5779d189ece9a31a0d526054513cc2", 
                             upload_file_path=audio_path)

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
    parser.add_argument("-o", "--output", type=str, default="./Audio Samples/ASR-Results.csv", help="Output path of the recognition result file in csv format.")
    args = parser.parse_args()

    speech_to_text(directory=args.directory, transcriptions=args.transcriptions, output=args.output)
