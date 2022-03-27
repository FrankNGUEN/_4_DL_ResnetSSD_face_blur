# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 01:39:08 2022

@author: HaiDuc
"""

import cv2
import os
import numpy as np
#--------------------------------------------------------------------------------------------------------------------
def load_face_models():
	# Link: https://github.com/LZQthePlane/Face-detection-base-on-ResnetSSD
	txt_file    = os.path.join("models", "deploy.prototxt.txt")
	weight_file = os.path.join("models", "res10_300x300_ssd_iter_140000.caffemodel")
	model       = cv2.dnn.readNet(txt_file, weight_file)
	return model
#--------------------------------------------------------------------------------------------------------------------
model  = load_face_models()
# Mo video tu file
#vid_file = "tests/test02.mp4"
#cam = cv2.VideoCapture(vid_file)
# Mở camera, số 0 là camera đầu tiên, nếu không mở được các bạn có thể thay bằng các số khác từ 1,2,3
cam = cv2.VideoCapture(0)
while True:
    # Đọc ảnh từ camera, tham số ret cho biết đọc thành công hay không
    ret, image = cam.read()
    (h, w) = image.shape[:2]
    # Đưa vào mạng để phát hiện khuôn mặt
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
    model.setInput(blob)
    detections = model.forward()
    # Lap qua ket qua dau ra
    for i in range(0, detections.shape[2]):
        # Lay confidence        
        confidence = detections[0, 0, i, 2]
        print(detections[0, 0, i])
        # Neu confiden > 0.5 moi xu ly
        if (detections[0, 0, i, 1] == 1) and (confidence > 0.5):
            # Lay toa do that
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            # Vẽ khung hình chữ nhật
            cv2.rectangle(image,(startX,startY),(endX,endY),(0,255,0),5)
    # Nếu đọc thành công
    if ret:
        # hiển thị lên màn hình
        cv2.imshow("Output", image)
        # nếu bấm q thì thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
#--------------------------------------------------------------------------------------------------------------------
cam.release()
cv2.destroyAllWindows()