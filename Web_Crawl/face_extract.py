import face_recognition
from PIL import Image,ImageDraw

image_path='data/without_mask/0.jpg'

face_image_np=face_recognition.load_image_file(image_path)
face_locations=face_recognition.face_locations(face_image_np) #model='hog' 기본값

#얼굴 영역의 좌표 ((top, right, bottom, left))
#print(face_locations)

face_image = Image.fromarray(face_image_np)
draw=ImageDraw.Draw(face_image)

#알아보기 쉽게 영역에서 값을가저옴
for face_locations in face_locations:
    top = face_locations[0]
    right = face_locations[1]
    bottom = face_locations[2]
    left = face_locations[3]
    draw.rectangle(((left, top),(right,bottom)),outline=(255,255,0),width=4)

face_image.show()

#마스크 인식

