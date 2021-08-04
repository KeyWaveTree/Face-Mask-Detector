# 08. keras_predict_data
# 06. keras_save_model

import tensorflow as tf     # 케라스 사용 위해
import matplotlib.pyplot as plt
import os

# 학습 데이터 생성
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    '../data/',
    validation_split= 0.2, # 검증/훈련 데이터 분리, 훈련데이터가 모의고사 문제지(답있는) 검증용 데이터는 수능 문제
    subset = 'training', # 훈련 데이터는 80%
    seed=123,
    image_size=(224,224),
    batch_size = 16,   # 연산량을 줄이기 위해 랜덤으로 덩어리 그룹을 만들어두는 것
)

# 검증 데이터 생성
valid_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    '../data',
    validation_split= 0.2,
    subset = 'validation',
    seed=123,
    image_size=(224,224),
    batch_size = 16,   # 연산량을 줄이기 위해 랜덤으로 덩어리 그룹을 만들어두는 것
)

# 모델을 위해서 추가로 가공 필요, 연속된 작업을 하나의 변수로 묶음
resize_and_crop = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomCrop(height=224, width = 224),
    tf.keras.layers.experimental.preprocessing.Rescaling(1./255)  #얘가 핵심 -> Rescaling 내에 들어오는 값을 모두 곱해줌, 실수로 인식
])

# resize 기능을 위의 각 데이터셋에 적용
rc_train_dataset = train_dataset.map(lambda x, y: (resize_and_crop(x), y)) # 문제와 정답이 x, y// 그리고 문제에만 resize and crop 적용)
rc_valid_dataset = valid_dataset.map(lambda x, y: (resize_and_crop(x), y))

# 모델 로드
model = tf.keras.models.load_model('../models/mymodel')

print(model.summary())

plt.figure(0)
plt.title('Valid Dataset Predict')
for images, labels in valid_dataset.take(1):
    rc_images = resize_and_crop(images)
    predict = model.predict(rc_images)
    print(predict)
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.imshow(images[i].numpy().astype('uint8'))
        if predict[i][0] > 0.5:
            predict_index = 1
        else:
            predict_index = 0

        if labels[i] == predict_index:
            result = 'OK'
        else:
            result = 'Wrong'

        plt.title(valid_dataset.class_names[predict_index] + '\n' + result)
        plt.axis('off')

plt.show()