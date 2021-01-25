from tencentcloud.common import credential
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException
from tencentcloud.asr.v20190614 import asr_client, models
import base64
import io
import sys
import json

if sys.version_info[0] == 3:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')


#本地文件方式请求
def tencent_asr(secret_id, secret_key, audio_path, language):
    try: 
        #重要，此处<Your SecretId><Your SecretKey>需要替换成客户自己的账号信息，获取方法：
        #https://cloud.tencent.com/product/asr/getting-started
        cred = credential.Credential(secret_id, secret_key) 
        httpProfile = HttpProfile()
        httpProfile.endpoint = "asr.tencentcloudapi.com"
        clientProfile = ClientProfile()
        clientProfile.httpProfile = httpProfile
        clientProfile.signMethod = "TC3-HMAC-SHA256"  
        client = asr_client.AsrClient(cred, "ap-shanghai", clientProfile) 
        #读取文件以及 base64
        #此处可以下载测试音频 https://asr-audio-1300466766.cos.ap-nanjing.myqcloud.com/test16k.wav
        with open(audio_path, "rb") as f:
            if sys.version_info[0] == 2:
                content = base64.b64encode(f.read())
            else:
                content = base64.b64encode(f.read()).decode('utf-8')
        #发送请求
        req = models.SentenceRecognitionRequest()
        params = {"ProjectId":0,"SubServiceType":2,"SourceType":1,"UsrAudioKey":"session-123"}
        req._deserialize(params)
        req.DataLen = len(content)
        req.Data = content
        if language == "Chinese":
            req.EngSerViceType = "16k_zh"
        else:
            req.EngSerViceType = "16k_en"
        req.VoiceFormat = "wav"
        resp = client.SentenceRecognition(req)
        print(resp.to_json_string())

        #windows 如果是 GBK 编码则用下面 print 语句替换上面 print 语句
        #print(resp.to_json_string().decode('UTF-8').encode('GBK') )

        return resp.Result

    except TencentCloudSDKException as err: 
        print(err)
        return ""


if __name__ == "__main__":
    tencent_asr(secret_id="AKIDVs2ze24Afy1ojor8MVNyHYaLp3IItTdK", secret_key="buyf8KmxmKbbKcdOsugBPKhtoagWrviG",
                audio_path="./Command-0.wav", language="Chinese")
