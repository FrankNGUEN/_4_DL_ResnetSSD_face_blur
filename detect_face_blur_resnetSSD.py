# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 00:34:28 2022
@author: HaiDuc
https://miai.vn/2021/12/03/phat-hien-va-lam-mo-mat-tu-dong-nhu-japan-av/
"""
import numpy as np
import cv2
import os
#--------------------------------------------------------------------------------------------------------------------
#
def pixelate_image(image, grid_size):
	(h, w)     = image.shape[:2]                                               # Chia anh thanh block x block o vuong
	xGridLines = np.linspace(0, w, grid_size + 1, dtype="int")
	yGridLines = np.linspace(0, h, grid_size + 1, dtype="int")
	for i in range(1, len(xGridLines)):                                        # Lap qua tung o vuong
		for j in range(1, len(yGridLines)):
			cell_startX = xGridLines[j - 1]                                    # Lay toa do cua o vuong hien tai
			cell_startY = yGridLines[i - 1]
			cell_endX   = xGridLines[j]
			cell_endY   = yGridLines[i]
			cell        = image[cell_startY:cell_endY, cell_startX:cell_endX]  # Trich vung anh theo toa do ben tren
			(B, G, R)   = [int(x) for x in cv2.mean(cell)[:3]]                 # Tinh trung binh cong vung anh va ve vao o vuong hien tai
			cv2.rectangle(image, (cell_startX, cell_startY), (cell_endX, cell_endY),(B, G, R), -1)
	return image
# De pat hien khuon mat, ta co the sd nhieu pp nhu Harrcascade, MTCNN, Dlib và SSD. Here, demo cách dùng pretrain ResnetSSD 
# SSD viết tắt của Single Shot Multibox Detector (tạm dịch: một phát ăn ngay trong công tác detection ). 
# Nó có nét tương đồng với YOLO (You Only Look Once). Nó chỉ thực một một pha duy nhất thay vì 2 pha như các thuật toán cũ hơn 
# chi tiết tại: https://miai.vn/2020/07/04/co-ban-ve-object-detection-voi-r-cnn-fast-r-cnn-faster-r-cnn-va-yolo
# Pretrain of  SSD
def load_face_models():
	# Link: https://github.com/LZQthePlane/Face-detection-base-on-ResnetSSD
	txt_file    = os.path.join("models", "deploy.prototxt.txt")
	weight_file = os.path.join("models", "res10_300x300_ssd_iter_140000.caffemodel")
	model       = cv2.dnn.readNet(txt_file, weight_file)
	return model
#--------------------------------------------------------------------------------------------------------------------
#Thứ nhất, Do kiến trúc mạng SSD input là 300,300 nên ảnh cần resize về kích thước đó trước khi đưa vào. Chú ý đoạn (300,300) trong lệnh blobfromImage
#Thứ hai, do mạng này pretrain với ImageNet nên chúng ta cũng phải mean subtraction để ảnh đầu vào của chúng ta tương tự như khi họ train. Đoạn này nè: (104.0, 177.0, 123.0). 
# https://pyimagesearch.com/2017/11/06/deep-learning-opencvs-blobfromimage-works/
#Thứ ba, mình chỉ xử lý với các predict có prob > 0.5 nhé. Ngưỡng này tuỳ các bạn cài đặt sao cho phù hợp với bài toán của các bạn.
# Load model
model = load_face_models()
cam   = cv2.VideoCapture(0)      # Mở camera, số 0 là camera đầu tiên, nếu không mở được các bạn có thể thay bằng các số khác từ 1,2,3                     
while True:
	ret, image = cam.read()      # Đọc ảnh từ camera, tham số ret cho biết đọc thành công hay không
	(h, w)     = image.shape[:2]        	                                                     # Load image
	blob       = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0)) 	         # Phat hien khuon mat
	model.setInput(blob)
	detections = model.forward()
	for i in range(0, detections.shape[2]):                                                      # Lap qua ket qua dau ra
		confidence = detections[0, 0, i, 2]                                                      # Lay confidence
		print(detections[0, 0, i])
		if (detections[0, 0, i, 1] == 1) and (confidence > 0.5):                                 # Neu confiden > 0.5 moi xu ly
			box                             = detections[0, 0, i, 3:7] * np.array([w, h, w, h])  # Lay toa do that
			(startX, startY, endX, endY)    = box.astype("int")		
			face                            = image[startY:endY, startX:endX]                    # Lay phan khuon mat
			face                            = pixelate_image(face, grid_size=int(w * 0.01))      # Pixelate
			image[startY:endY, startX:endX] = face                                               # Ve de phan pixelate len
			cv2.imshow("Output", image)                                                          # Hien thi len man hinh
	if cv2.waitKey(1) & 0xFF == ord('q'):                                                        # An q la thoat
		break
cam.release()
cv2.destroyAllWindows()                                                                          # Destroy all the windows
#--------------------------------------------------------------------------------------------------------------------