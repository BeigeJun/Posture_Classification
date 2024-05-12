import cv2
import os
directory = '/DATASET/Sit'

if not os.path.exists(directory):
    os.makedirs(directory)

cap = cv2.VideoCapture(0)

count = 0
directory = 'C:/Users/wns20/PycharmProjects/SMART_CCTV/DATASET' + '/Sit'

while True:

    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow('Pose Detection', frame)

    key = cv2.waitKey(1)
    if key == ord('c'):
        file_name = f'{directory}/img{count}.jpg'
        frame = cv2.flip(frame, 1)
        cv2.imwrite(file_name, frame)
        count += 1
    elif key == ord('q'):
        break
