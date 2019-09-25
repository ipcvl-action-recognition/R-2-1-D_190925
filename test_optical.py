# from networks import *
# import models
import cv2
import torch
import numpy as np
import time

video_file = 'D:/fire1.mp4'
cap = cv2.VideoCapture(video_file)
ret, frame1 = cap.read()
prev = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[...,1] = 255
while cap.isOpened():
    ret, frame2 = cap.read()
    if ret:
        next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        start = time.time()
        flow = cv2.calcOpticalFlowFarneback(prev, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[:, :, 0], flow[:, :, 1])
        hsv[:, :, 0] = ang*180/np.pi/2
        hsv[:, :, 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        end = time.time()
        # print("time : {0:0.2f}".format(end-start), "second")
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        cv2.imshow("frame", rgb)
        cv2.imshow("hsv", hsv)
        cv2.waitKey(1)
        prev = next
    else:
        break
cap.release()
