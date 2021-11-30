import http.client
import json

from aliyunsdkcore.client import AcsClient
from aliyunsdkcore.request import CommonRequest


def alibaba(app_key, access_key_id, access_key_secret, audio_path):
    appKey = app_key

    client = AcsClient(access_key_id, access_key_secret, "cn-shanghai")

    # 创建request，并设置参数。
    request = CommonRequest()
    request.set_method('POST')
    request.set_domain('nls-meta.cn-shanghai.aliyuncs.com')
    request.set_version('2019-02-28')
    request.set_action_name('CreateToken')
    response = client.do_action_with_exception(request)

    token = str(response, 'utf-8')
    token = json.loads(token)
    token = token['Token']['Id']

    # 服务请求地址
    url = 'http://nls-gateway.cn-shanghai.aliyuncs.com/stream/v1/asr'

    # 音频文件
    audioFile = audio_path
    format = 'wav'
    sampleRate = 16000
    enablePunctuationPrediction  = True
    enableInverseTextNormalization = True
    enableVoiceDetection  = False

    # 设置RESTful请求参数
    request = url + '?appkey=' + appKey
    request = request + '&format=' + format
    request = request + '&sample_rate=' + str(sampleRate)

    if enablePunctuationPrediction :
        request = request + '&enable_punctuation_prediction=' + 'true'

    if enableInverseTextNormalization :
        request = request + '&enable_inverse_text_normalization=' + 'true'

    if enableVoiceDetection :
        request = request + '&enable_voice_detection=' + 'true'

    print('Request: ' + request)

    asr_result = process(request, token, audioFile)

    return asr_result


def process(request, token, audioFile) :
    # 读取音频文件
    with open(audioFile, mode = 'rb') as f:
        audioContent = f.read()

    host = 'nls-gateway.cn-shanghai.aliyuncs.com'

    # 设置HTTP请求头部
    httpHeaders = {
        'X-NLS-Token': token,
        'Content-type': 'application/octet-stream',
        'Content-Length': len(audioContent)
        }

    # Python 2.x使用httplib
    # conn = httplib.HTTPConnection(host)

    # Python 3.x使用http.client
    conn = http.client.HTTPConnection(host)

    conn.request(method='POST', url=request, body=audioContent, headers=httpHeaders)

    response = conn.getresponse()
    print('Response status and response reason:')
    print(response.status ,response.reason)

    body = response.read()
    try:
        print('Recognize response is:')
        body = json.loads(body)
        print(body)

        status = body['status']
        if status == 20000000 :
            result = body['result']
            print('Recognize result: ' + result)
        else :
            print('Recognizer failed!')
            result = ""

    except ValueError:
        print('The response is not json format string')
        result = ""

    conn.close()

    return result
