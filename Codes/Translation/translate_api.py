import requests
import json
import os
import sys
import urllib.request

def send_papago_api(text):
    client_id = "b9slkUpO1WtvFlRBuyfn" # 개발자센터에서 발급받은 Client ID 값
    client_secret = "B1t3eWyHwi" # 개발자센터에서 발급받은 Client Secret 값
    
    data = "source=ko&target=en&text=" + encText
    url = "https://openapi.naver.com/v1/papago/n2mt"
    request = urllib.request.Request(url)
    request.add_header("X-Naver-Client-Id",client_id)
    request.add_header("X-Naver-Client-Secret",client_secret)
    response = urllib.request.urlopen(request, data=data.encode("utf-8"))
    rescode = response.getcode()
    if(rescode==200):
        response_body = response.read()
        print(response_body.decode('utf-8'))
        return response_body.decode('utf-8')
    else:
        print("Error Code:" + rescode)
        return ""


        
    
encText = urllib.parse.quote("반갑습니다")
send_papago_api(encText)
    
  

# 호출 예시
#send_api("/test", "POST")
origin = "Hello. Nice to meet you."
data = {"source": "en", "target": "ko", "text": origin}
#response = send_api("POST", data)

#print(response)