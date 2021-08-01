# 사람 이미지 다운로드
from urllib.request import Request, urlopen
import json
import os

#url 링크
api_url = 'https://api.github.com/repos/prajnasb/observations/contents/experiements/data/without_mask?ref=master'

#http 웹페이지에 접속할 때 필요한 유저 에이전트 정보
hds = {'User-Agent': 'Mozilla/5.0'}

request = Request(api_url, headers=hds)#http requset, from urllib.request import Request
response = urlopen(request)#from urllib.request import urlopen
directory_bytes = response.read() #내용을 하나의 문자열로 읽어온다
directory_str = directory_bytes.decode('utf-8')
contents = json.loads(directory_str)

for i in range(len(contents)):
    content = contents[i]
    request = Request(content['download_url'])
    response = urlopen(request)
    image_data = response.read()

    if not os.path.exists('Data'):
        os.mkdir('Data')
    if not os.path.exists('Data/without_mask'):
        os.mkdir('Data/without_mask')

    image_file = open('Data/without_mask/' + content['name'], 'wb')
    image_file.write(image_data)
    print('다운로드 완료(' + str(i + 1) + '/' + str(len(contents)) + '): ' + content['name'])
    break
