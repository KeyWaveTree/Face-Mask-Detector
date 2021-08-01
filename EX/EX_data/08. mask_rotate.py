import face_recognition
import numpy as np
from PIL import Image, ImageDraw

face_image_path = 'Data/without_mask/1.jpg'
mask_image_path = 'Data/mask.png'

face_image_np = face_recognition.load_image_file(face_image_path)
face_locations = face_recognition.face_locations(face_image_np)
face_landmarks = face_recognition.face_landmarks(face_image_np, face_locations)

face_landmark_image = Image.fromarray(face_image_np)
draw = ImageDraw.Draw(face_landmark_image)
mask_image = Image.open(mask_image_path)

for face_landmark in face_landmarks:
    # 얼굴의 축(좌표)의 역할
    nb = face_landmark['nose_bridge']
    nb_top = nb[0]
    nb_bottom = nb[3]
    dx = nb_bottom[0] - nb_top[0]
    dy = nb_bottom[1] - nb_top[1]

    #아크탄젠트로 라디안 값에서 각도를 구해야 한다.
    face_radian = np.arctan2(dy, dx)
    face_degree = np.rad2deg(face_radian)

    mask_degree = 90 - face_degree

    mask_image = mask_image.resize((80, 50))
    mask_image = mask_image.rotate(mask_degree)

    face_landmark_image.paste(mask_image, (0, 0), mask_image)

face_landmark_image.show()
