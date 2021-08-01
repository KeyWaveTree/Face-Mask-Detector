from urllib.request import Request, urlopen
import json
import os

# def image_download(url, filepath):
#     request = Request(url)
#     response=urlopen(request)
#     image_data=response.read()
#     file =open(filepath,'wb')
#     file.write(image_data)
#     file.close()
#     print(url+'로 부터', +filepath + '에 다운로드 완료')
#
# mask_url='https://github.com/prajnasb/observations/raw/master/mask_classifier/Data_Generator/images/blue-mask.png'
# image_download(mask_url, 'Data/mask.png')
# exit()

api_url = 'https://api.github.com/repos/prajnasb/observations/contents/experiements/data/without_mask?ref=master'

hds = {'User-Agent': 'Mozilla/5.0'}

request = Request(api_url, headers=hds)
response = urlopen(request)
directory_bytes = response.read()
directory_str = directory_bytes.decode('utf-8')

contents = json.loads(directory_str)

download_number = int(input('%d장 중 어느 사진부터 다운로드를 하면 될까요? 숫자를 입력해 주세요(최대:0~686):' % len(contents)))
if download_number==('' or 0):
    download_number=0
elif download_number > len(contents):
    print('숫자를 초과하였습니다.')
    exit(0)

for i in range(download_number,len(contents)):
    download_continue=''
    content = contents[i]
    # print(content['download_url'])
    request = Request(content['download_url'])
    response = urlopen(request)
    image_data = response.read()

    if not os.path.exists('data'):
        os.mkdir('data')
    if not os.path.exists('data/without_mask'):
        os.mkdir('data/without_mask')

    image_file = open('Data/without_mask/' + content['name'], 'wb')
    image_file.write(image_data)
    print('다운로드 완료(' + str(i + 1) + '/' + str(len(contents)) + '): ' + content['name'])

    download_continue = input('계속 다운로드? (Y/N):')
    type(download_number)
    if download_continue == 'Y' or download_continue=='y':
        continue
    else:
        break
