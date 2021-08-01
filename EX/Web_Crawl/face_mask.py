import face_recognition
from PIL import Image, ImageDraw

#랜드마크를 이용한 마스크 합성
face_image_path = 'data/without_mask/1.jpg'

face_image = face_recognition.load_image_file(face_image_path)
# face_locations = face_recognition.face_locations(face_image_np)
# face_landmarks = face_recognition.face_landmarks(face_image_np, face_locations)
#
# face_landmark_image = Image.fromarray(face_image_np)
# draw = ImageDraw.Draw(face_landmark_image)
#
# face_landmark_image.show()