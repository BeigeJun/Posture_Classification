import csv
import cv2
import os
from PIL import Image
import torch
from torchvision import models
import torchvision.transforms as t
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
import cv2

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
video_path = 'C:/Users/wns20/PycharmProjects/SMART_CCTV/output.avi'

fourcc = cv2.VideoWriter_fourcc(*'XVID')

out = cv2.VideoWriter(video_path, fourcc, 20.0, (640, 480))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow('Video', frame)

    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()