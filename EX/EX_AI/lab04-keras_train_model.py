# 04. keras_train_model.py
import tensorflow as tf
import matplotlib.pyplot as plt

# dense -> 완전연결신경망
# parameter -> 그래프를 그리기 위해 필요한 값, ex) 선형회귀의 경우 2개 필요함
# epochs -> 구슬이 몇번 내려가게 할 것인가
# batch_size 따라 step수 변화
# epoch와 batch_size - https://m.blog.naver.com/qbxlvnf11/221449297033
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    '../Data/',
    validation_split=0.2,
    subset='training',
    seed=123,
    image_size=(224, 224),
    batch_size=16
)

#수능 본시험 느낌 #평가 단계
valid_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    '../Data/',
    validation_split=0.2,
    subset='validation',
    seed=123,
    image_size=(224, 224),
    batch_size=16
)

# “Sequential” -> 이어주는 역할
resize_and_crop = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomCrop(height=224, width=224),
    tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255)
])

rc_train_dataset = train_dataset.map(lambda x, y: (resize_and_crop(x), y))
rc_valid_dataset = valid_dataset.map(lambda x, y: (resize_and_crop(x), y))

# 모델 생성
model = tf.keras.applications.MobileNet(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

# “model.trainable = False” -> imagenet train dataset 값 변경 불가능하게 함
model.trainable = False

model = tf.keras.Sequential([
    model,
    # “Pooling” -> 연산량 감소하게 함
    tf.keras.layers.GlobalAveragePooling2D(),#“AveragePooling” -> 대표값을 평균을 내어 연산량 감소하게 함(그러면서 의미는 포함하고 있음)
    tf.keras.layers.Dense(1)# “Dense(1)” -> 값이 하나로 나오게 함
])

print(model.summary())

# 모델 학습
learning_rate = 0.001
model.compile(
    # “BinaryCrossentropy” -> 단순하게 설계된 Crossentropy, 분류할 때 사용
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    # “optimizer” -> loss값을 보고 이에 따라서 튜닝을 하기 위한것
    # “optimizers.RMScrop(lr)” -> “lr = Learning Rate” -> 속도(작으면 느려지고 크면 빨라짐)
    optimizer=tf.keras.optimizers.RMSprop(lr=learning_rate),
    metrics=['accuracy']# “metrics=[‘accuracy’]” -> 정확도 보는 옵션(?)
)

history = model.fit(rc_train_dataset, epochs=2, validation_data=rc_valid_dataset)

print(history)