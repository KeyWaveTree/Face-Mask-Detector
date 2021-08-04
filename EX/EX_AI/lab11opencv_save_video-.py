# 11. opencv_save_video.py
import tensorflow as tf
import numpy as np
import cv2
import os

#전처리 할 때 필요하니 그대로
resize_and_crop = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomCrop(height=224, width=224),
    tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
])
#10번코드 그대로 모델을 불러옴
#마스크 쓰고 있던 먹고 있던 얼굴을 찾을 수 있는 모델
face_mask_recognition_model = cv2.dnn.readNet(
    '../models/face_mask_recognition.prototxt',
    '../models/face_mask_recognition.caffemodel'
)
#마이 모델
mask_detector_model = tf.keras.models.load_model('../models/mymodel')

cap = cv2.VideoCapture('../data/04.mp4')
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

if not os.path.exists('../outputs'):
    os.mkdir('../outputs')

out = None

#cap.read이용해 영상 프레임이 남아있는지 확인+프레임 안에 데이터를 받아옴(while문 속 코드는 9번과 동일함)
while cap.isOpened():
    ret, image = cap.read()
    if not ret:
        break

    height, width = image.shape[:2] #0부터 1까지

    blob = cv2.dnn.blobFromImage(image, scalefactor=1., size=(300, 300), mean=(104., 177., 123.)) #한 프레임 이미지를 여러 개의 덩어리로 분리
    face_mask_recognition_model.setInput(blob) #이 모델에서 blob단위로 보며 확률이 높은 위치의 네모 박스를 보기를 요구
    face_locations = face_mask_recognition_model.forward() #forward라는 함수 호출 (predict함수 기능과 비슷)

    result_image = image.copy()

    for i in range(face_locations.shape[2]):
        confidence = face_locations[0, 0, i, 2] #좀 더 정교하게 하기 위해 확률값이 들어가 있음
        if confidence < 0.5: #값을 작게할수록 얼굴에 민감하게 반응함 (하지만 얼굴이 아닌 것도 처리가 될 수 있음) (값을 올리면 얼굴에 대한 기준이 엄격해짐)
            continue

        left = int(face_locations[0, 0, i, 3] * width)
        top = int(face_locations[0, 0, i, 4] * height)
        right = int(face_locations[0, 0, i, 5] * width)
        bottom = int(face_locations[0, 0, i, 6] * height)

        face_image = image[top:bottom, left:right]
        face_image = cv2.resize(face_image, dsize=(224, 224))
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

        rc_face_image = resize_and_crop(np.array([face_image]))

        predict = mask_detector_model.predict(rc_face_image)
        if predict[0][0] > 0.5:
            color = (0, 0, 255) #BGR형태로 있음 ->빨강
            label = 'without_mask'
        else:
            color = (0, 255, 0) #초록
            label = 'with_mask'

        cv2.rectangle(
            result_image,
            pt1=(left, top),
            pt2=(right, bottom),
            thickness=2,
            color=color,
            lineType=cv2.LINE_AA
        )

        cv2.putText(
            result_image,
            text=label,
            org=(left, top - 10),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.8,
            color=color,
            thickness=2,
            lineType=cv2.LINE_AA
        )
    if out is None:
        out = cv2.VideoWriter('../outputs/output.wmv', fourcc, cap.get(cv2.CAP_PROP_FPS),
                              (image.shape[1], image.shape[0]))
    else:
        out.write(result_image)
    #프레임 가져와서 진행률도 같이 띄움
    cv2.imshow('result', result_image)
    if cv2.waitKey(1) == ord('q'):
        break

out.release()
cap.release()