# 09. predict_from_image.py
import face_recognition
from PIL import Image, ImageDraw
import tensorflow as tf
import numpy as np

#가공해주는 작업
resize_and_crop = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomCrop(height=224, width=224), #자르기
    tf.keras.layers.experimental.preprocessing.Rescaling(1./255) #rescaling 1/255을 곱해줌
])

#6번 코드에서 불러옴 models항목 안의 load_model에서 경로를 불러옴 이전에 학습해놓은 모델을 그대로 불러옴
model = tf.keras.models.load_model('../models/mymodel')
#마스크 안쓴 이미지 불러옴
face_image_path = '../../Data/without_mask/0.jpg'
#
face_image_np = face_recognition.load_image_file(face_image_path)
face_locations = face_recognition.face_locations(face_image_np)  #얼굴 네모 박스 영역을 모두 가져옴
#최종 이미지-face_image_np에서 받아옴
face_image = Image.fromarray(face_image_np)
draw = ImageDraw.Draw(face_image) #draw변수로 face이미지에 예측 결과를 그려줌
#얼굴 하나하나를 돌아야 하기 때문에 하나씩 받아오게 만듬
for face_location in face_locations:
    top = face_location[0]
    right = face_location[1]
    bottom = face_location[2]
    left = face_location[3]
    face_crop = face_image.crop((left - 10, top - 10, right + 10, bottom + 10)) #얼굴이 타이트하게 잘려서 여유공간을 줌
    face_crop = face_crop.resize((224, 224)) #이미지 크기 가공
    face_crop_np = np.array(face_crop) #이미지 형태를 배열 형태로 변환 (픽셀값 3차원으로)
    rc_face_crop = resize_and_crop(np.array([face_crop_np])) #모델이 여러 개를 한번에 처리할 수 있어서 하나의 배열로 묶음
    predict = model.predict(rc_face_crop)
    if predict[0][0] > 0.5:
        label = 'without_mask'
    else:
        label = 'with_mask'
    #결과 그려줌
    draw.rectangle(((left, top), (right, bottom)), outline=(255, 255, 0), width=4) #초록박스
    draw.text((left, top - 10), label)
#마스크 쓴 사람 얼굴 데이터는 인식자체를 할 수 없다는 한계점이 있음
face_image.show()