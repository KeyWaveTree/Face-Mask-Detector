# 06. keras_save_model.py
# 04. keras_train_model.py
import tensorflow as tf
import matplotlib.pyplot as plt
import os

train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    '../Data/',
    validation_split=0.2,  # 검증/훈련 데이터 분리, 훈련데이터가 모의고사 문제지(답있는) 검증용 데이터는 수능 문제
    subset='training',  # 훈련 데이터는 80%
    seed=123,
    image_size=(224, 224),
    batch_size=16,  # 연산량을 줄이기 위해 랜덤으로 덩어리 그룹을 만들어두는 것
)

valid_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    '../Data/',
    validation_split=0.2,
    subset='validation',
    seed=123,
    image_size=(224, 224),
    batch_size=16
)

# 모델을 위해서 추가로 가공 필요, 연속된 작업을 하나의 변수로 묶음
resize_and_crop = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomCrop(height=224, width = 224),
    tf.keras.layers.experimental.preprocessing.Rescaling(1./255)  #얘가 핵심 -> Rescaling 내에 들어오는 값을 모두 곱해줌, 실수로 인식
])

# resize 기능을 위의 각 데이터셋에 적용
rc_train_dataset = train_dataset.map(lambda x, y: (resize_and_crop(x), y))#문제와 정답이 x, y 그리고 문제에만 resize and crop 적용)
rc_valid_dataset = valid_dataset.map(lambda x, y: (resize_and_crop(x), y))

# 모델 생성
model = tf.keras.applications.MobileNet( # MobileNet: 모바일 기기에서도 동작할수있을만큼 가벼운 모델, cpu로만 학습해도 할만한 ..
    input_shape= (224, 224, 3),# 어떤 모델의 입력값의 형태를 지정해주는 역할, 3은 rgb 채널
    include_top=False, # 모델의 처음과 끝을 입맛에 맞춰 바꿀수있음, 출력층 없애버리고 우리가 원하는 층으로 대체할 것임
    weights= 'imagenet'#사전학습된 튜닝값을 받아오겠다.
)

print(model.summary())
model.trainable = False   # 지금 튜닝된 값은 학습을 하게 만들지 않겠다, 이후 값들만 학습하겠다.
print(model.summary())

model = tf.keras.Sequential([
    model,
    tf.keras.layers.GlobalAveragePooling2D(),#평균을 내는 것(여러개의 값을 하나의 값으로 통일해서 데이터의 전체 크기를 줄이는 역할)
    tf.keras.layers.Dense(1)    # 최종적으로는 한개의 값만 출력
])

print(model.summary())

# 모델 학습(평가함수와 알고리즘 선택이 필요)
learning_rate = 0.001
model.compile(
    # crossentropy: '분류 문제를 풀때 binary로 사용하는 평가함수(분류가 2개짜리일때)'
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True), # 모델 성능이 얼만큼 올라와있는가,
    optimizer=tf.keras.optimizers.RMSprop(lr=learning_rate),  #lr:learning rate(얼마나 공을 빠르게 떨어뜨릴건가)
    #----- 여기까지가 필수조건들이었음 다음줄부터는 부가라인
    metrics = ['accuracy'],
)

# 에포크 속에 step이 있음 -> batch size만큼씩 학습한번하는게 1 step-> batch size 32로 늘리면 step이 절반으로 줄 것임
# batch size는 적당해야함
history = model.fit(rc_train_dataset, epochs=2, validation_data=rc_valid_dataset)    # 에포크: 훈련을 몇번할 것인가
print(history)
