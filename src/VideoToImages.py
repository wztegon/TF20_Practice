import cv2
import os

def getFrame(videoPath, svPath):
	cap = cv2.VideoCapture(videoPath)
	numFrame = 0
	coutnum = 0
	flagnum = 0
	while True:
		if cap.grab():
			flag, frame = cap.retrieve()
			if not flag:
				flagnum += 1
				if flagnum > 10:
					print('完成图片提取')
					break
				else:
					continue
			else:
				# cv2.imshow('video', frame)
				numFrame += 1
				
				if (numFrame > 25*40 and numFrame < 25*45) or\
					(numFrame > 25*100 and numFrame < 25*100) or\
					(numFrame > 25*150 and numFrame < 25*160):
					coutnum += 1
					newPath = os.path.join(svPath, str(coutnum) + ".jpg")
					cv2.imencode('.jpg', frame)[1].tofile(newPath)
		if cv2.waitKey(10) == 27:
			break

videoPath = r'C:\Users\Administrator\Desktop\auto.mp4'
savePicturePath = r'C:\Users\Administrator\Desktop\autoImages'
getFrame(videoPath, savePicturePath)