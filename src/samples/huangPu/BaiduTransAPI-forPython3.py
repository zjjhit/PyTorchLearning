# 百度通用翻译API,不包含词典、tts语音合成等资源，如有相关需求请联系translate_api@baidu.com
# coding=utf-8

import hashlib
import http.client
import json
import random
import urllib

appid = '20200903000558136'  # 填写你的appid
secretKey = 'RtS_uVMx_6hZIQBZttmS'  # 填写你的密钥

httpClient = None
myurl = '/api/trans/vip/translate'

fromLang = 'auto'  # 原文语种
toLang = 'zh'  # 译文语种
salt = random.randint(32768, 65536)
q = 'apple'
sign = appid + q + str(salt) + secretKey
sign = hashlib.md5(sign.encode()).hexdigest()
myurl = myurl + '?appid=' + appid + '&q=' + urllib.parse.quote(q) + '&from=' + fromLang + '&to=' + toLang + '&salt=' + str(
    salt) + '&sign=' + sign

try:
    print(myurl)
    httpClient = http.client.HTTPConnection('api.fanyi.baidu.com')

    headerdata = {
        "Host": "api.fanyi.baidu.com",
        "Accept-Encoding": "gzip",
        "User-Agent": "Android-ALI-Moblie 1.3.0",
        "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8",
        "Connection": "Keep-Alive"}

    httpClient.request('GET', myurl)

    # response是HTTPResponse对象
    response = httpClient.getresponse()
    result_all = response.read().decode("utf-8")
    result = json.loads(result_all)

    print(result)

# except Exception as e:
#     print(e)
finally:
    if httpClient:
        httpClient.close()
