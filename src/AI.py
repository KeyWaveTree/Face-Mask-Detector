import tensorflow as tf
import numpy as np
import os

resize_and_crop = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomCrop(height=224, width=224),#랜덤크롭
    tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255)
])

#학습 데이터 로드 -데이터 수집, 가공은 이미 했기때문에 로드만 해준다.
def load_data():
    face_train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        'Data/',
        validation_split=0.2,
        subset='training',
        seed=123,
        image_size=(224, 224),
        batch_size=16
    )#모의고사

    face_valid_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        'Data/',
        validation_split=0.2,
        subset='validation',
        seed=123,
        image_size=(224, 224),
        batch_size=16
    )#평가 문제

    rc_train_dataset = face_train_dataset.map(lambda x, y: (resize_and_crop(x), y))
    rc_valid_dataset = face_valid_dataset.map(lambda x, y: (resize_and_crop(x), y))

    return rc_train_dataset, rc_valid_dataset


# 모델 생성
# 저장된 모델이 있으면 가져오고 없으면 그때 생성하게 함
def create_model():
    if os.path.exists('Data/Models/mymodel'):
        model = tf.keras.models.load_model('Data/Models/mymodel') #내 모델 있다면 그대로 불러옴

        model.layers[0].trainable = False #학습이 가능하게 만들것인가
        model.layers[2].trainable = True  #없어도 상관 없는데 불러와서 더 학습할 때 (중간 상태 저장해놓고 이어하기)를 위해서 씀
    else:
        model = tf.keras.applications.MobileNet(
            input_shape=(224, 224, 3),
            include_top=False,
            weights='imagenet'
        )

        model.trainable = False

        model = tf.keras.Sequential([
            model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(1)
        ])

        learning_rate = 0.001
        model.compile(
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            optimizer=tf.keras.optimizers.RMSprop(lr=learning_rate),
            metrics=['accuracy']
        )

        face_train_dataset, face_valid_dataset = load_data()
        train_model(model, 20, face_train_dataset, face_valid_dataset, True)
    return model


# 모델 학습
#(학습할)모델,몇번할지,학습을 위한 트레인 데이터셋, 시험을 위한 데이터 셋,저장할지말지 결정하는 모델로 5개의 인자 받아옴
def train_model(model, epochs, face_train_dataset, face_valid_dataset, save_model):
    history = model.fit(face_train_dataset, epochs=epochs, validation_data=face_valid_dataset)
    if save_model:
        model.save('Data/Models/mymodel')
    return history


# 학습된 모델로 예측
def predict(model, image):
    rc_image = resize_and_crop(np.array([image]))#np: 여러 이미지 동시에 예측가능하게 설계돼서 하나만 넣어도 np배열로 만들어서 넣어줘야 함
    result = model.predict(rc_image) ##최종 가공된 데이터를 rc_image에 넣어줌
    if result[0] > 0.5:
        return 1
    else:
        return 0


if __name__ == '__main__':
    face_train_dataset, face_valid_dataset = load_data()
    model = create_model()
    train_model(model, 2, face_train_dataset, face_valid_dataset, True) #학습 누적된 모델 생성