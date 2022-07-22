import imutils
import numpy as np
import AI as ai
import cv2
import os

from imutils.video import VideoStream
#마스크 디텍터 모델: 마스크를 썼는지 않썼는지 확률을 반환하는 모델
#페이스 마스크 모델: 마스크를 쓰던 안쓰던 얼굴 영역을 반환해 주는 모델

# 영상 처리
def video_processing(video_path, background):
	face_mask_recognition_model = cv2.dnn.readNet(
		'Data/Models/face_mask_recognition.prototxt',
		'Data/Models/face_mask_recognition.caffemodel'
	)

	mask_detector_model = ai.create_model()

	cap = cv2.VideoCapture(video_path)
	fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

	frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	now_frame = 1

	if not os.path.exists('outputs'):
		os.mkdir('outputs')

	out = None

	colors = [(0, 255, 0), (0, 0, 255)] # 마스크0(0)->초록 마스크x(1)->빨강 컬러값 (BGR순서임)
	labels = ['with_mask', 'without_mask']

	while cap.isOpened():
		ret, image = cap.read()
		if not ret:
			break

		height, width = image.shape[:2]

		blob = cv2.dnn.blobFromImage(image, scalefactor=1., size=(300, 300), mean=(104., 177., 123.))
		face_mask_recognition_model.setInput(blob)

		# 마스크를 쓰던 안쓰던 얼굴 위치를 반환
		face_locations = face_mask_recognition_model.forward()

		result_image = image.copy()

		for i in range(face_locations.shape[2]):
			confidence = face_locations[0, 0, i, 2]
			if confidence < 0.5:
				continue

			left = int(face_locations[0, 0, i, 3] * width)
			top = int(face_locations[0, 0, i, 4] * height)
			right = int(face_locations[0, 0, i, 5] * width)
			bottom = int(face_locations[0, 0, i, 6] * height)

			face_image = image[top:bottom, left:right]
			face_image = cv2.resize(face_image, dsize=(224, 224))
			face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

			predict = ai.predict(mask_detector_model, face_image)

			cv2.rectangle(
				result_image,
				pt1=(left, top),
				pt2=(right, bottom),
				thickness=2,
				color=colors[predict],
				lineType=cv2.LINE_AA
			)

			cv2.putText(
				result_image,
				text=labels[predict],
				org=(left, top - 10),
				fontFace=cv2.FONT_HERSHEY_SIMPLEX,
				fontScale=0.8,
				color=colors[predict],
				thickness=2,
				lineType=cv2.LINE_AA
			)

		if out is None:
			out = cv2.VideoWriter(
				'outputs/output.wmv',
				fourcc,
				cap.get(cv2.CAP_PROP_FPS),
				(image.shape[1], image.shape[0])
			)
		else:
			out.write(result_image)

		# (10/400): 11%
		print('(' + str(now_frame) + '/' + str(frame_count) + '): ' + str(now_frame * 100 // frame_count) + '%')
		now_frame += 1

		if not background:
			cv2.imshow('result', result_image)
			if cv2.waitKey(1) == ord('q'):
				break

	out.release()
	cap.release()

def camera_processing(camera_number, background):
		face_mask_recognition_model = cv2.dnn.readNet(
			'Data/Models/face_mask_recognition.prototxt',
			'Data/Models/face_mask_recognition.caffemodel'
		)

		mask_detector_model = ai.create_model()

		vs = imutils.video.VideoStream(src=camera_number).start()

		if not os.path.exists('outputs'):
			os.mkdir('outputs')

		colors = [(0, 255, 0), (0, 0, 255)]  # 마스크0(0)->초록 마스크x(1)->빨강 컬러값 (BGR순서임)
		labels = ['with_mask', 'without_mask']

		while True:
			frame = vs.read()

			height, width = frame.shape[:2]
			blob = cv2.dnn.blobFromImage(frame, scalefactor=1., size=(300, 300), mean=(104., 177., 123.))
			face_mask_recognition_model.setInput(blob)

			face_locations = face_mask_recognition_model.forward()

			result_frame = frame.copy()

			for i in range(face_locations.shape[2]):
				confidence = face_locations[0, 0, i, 2]
				if confidence > 0.5:
					box = face_locations[0, 0, i, 3:7] * np.array([width, height, width, height])
					(left, top, right, bottom) = box.astype("int")
					(left, top) = (max(0, left), max(0, top))
					(right, bottom) = (min(width - 1, right), min(height - 1, bottom))

					face_frame = frame[left:right, top:bottom]
					face_frame = cv2.resize(face_frame, dsize=(224, 224))
					face_frame = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)

					predict = ai.predict(mask_detector_model, face_frame)

					cv2.rectangle(
						result_frame,
						pt1=(left, top),
						pt2=(right, bottom),
						thickness=2,
						color=colors[predict],
						lineType=cv2.LINE_AA
					)

					cv2.putText(
						result_frame,
						text=labels[predict],
						org=(left, top - 10),
						fontFace=cv2.FONT_HERSHEY_SIMPLEX,
						fontScale=0.8,
						color=colors[predict],
						thickness=2,
						lineType=cv2.LINE_AA
					)

			if not background:
				cv2.imshow('result', result_frame)
				if cv2.waitKey(1) == ord('q'):
					break

		cv2.destroyAllWindows()
		vs.stop()

if __name__ == '__main__':
	camera_processing(1,True)
# 과적합되면 오히려 부작용이 생김
# (2번씩 계속 학습해서 모의고사 문제를 다 외워버리는 수준이 되면 모의고사 문제는 잘 풀지만 나머지는 잘 못푸는 현상이 발생할 수 있음)