# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         translate
# Description:  
# Author:       lenovo
# Date:         2020/9/3
# -------------------------------------------------------------------------------

import hashlib
import http.client
import json
import random
import time
import urllib


def baidu_translate(q, fromLang, to_lang):
    # 开发者信息申请信息处的APP ID
    appid = '20200903000558136'
    # 开发者信息申请信息处的密钥
    secretKey = 'RtS_uVMx_6hZIQBZttmS'
    # 本机的ip地址
    httpClient = None
    myurl = '/api/trans/vip/translate'

    salt = random.randint(32768, 65536)
    sign = appid + q + str(salt) + secretKey
    sign = hashlib.md5(sign.encode()).hexdigest()
    myurl = myurl + '?appid=' + appid + '&q=' + urllib.parse.quote(q) + '&from=' + fromLang + '&to=' + to_lang + '&salt=' + str(
        salt) + '&sign=' + sign

    try:
        httpClient = http.client.HTTPConnection('api.fanyi.baidu.com')
        httpClient.request('GET', myurl)

        # response是HTTPResponse对象
        response = httpClient.getresponse()
        result_all = response.read().decode("utf-8")
        result = json.loads(result_all)

        print(result)
        return result['trans_result'][0]['dst']

    except Exception as e:
        print('T')
        print(e)
    finally:
        if httpClient:
            httpClient.close()


def back_translate(content, from_lang, language_types):
    results = []
    for to_lang in language_types:
        try:
            # 将原始语种翻译成另外一种语言
            temp_content = baidu_translate(content, from_lang, to_lang)
            time.sleep(1)
            # 往回翻译
            result_content = baidu_translate(temp_content, to_lang, from_lang)
            # 存储翻译结果
            results.append(result_content)
        except Exception as  e:
            print("Error", e)
    return results


def run(path_):
    fin = open(path_, 'r')
    fout = open(path_ + '.log', 'w')
    for one in fin:
        tmp_ = one.rstrip().split('\t')
        r_ = back_translate(tmp_[2], 'zh', 'en')
        fout.write(one.rstrip() + '\t\t' + '  '.join(r_) + '\n')
    fout.close()
    fin.close()


import sys

if __name__ == '__main__':
    # lang = ['en']
    # content = "该地方填写需要翻译的内容"
    # results = back_translate(content, "zh", lang)
    # print(results)

    run(sys.argv[1])
