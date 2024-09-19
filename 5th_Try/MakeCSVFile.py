import csv
import cv2
import os
import torch
from torchvision import models
import torchvision.transforms as t

video_path = 'C:/Users/wns20/PycharmProjects/SMART_CCTV/5th_Try/Data/output.avi'
cap = cv2.VideoCapture(video_path)
csv_file = 'C:/Users/wns20/PycharmProjects/SMART_CCTV/5th_Try/Data/pos_data.csv'

fieldnames = []
for i in range(17):
    fieldnames.append(f'keypoint_{i + 1}_x')
    fieldnames.append(f'keypoint_{i + 1}_y')
fieldnames.append('label')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.detection.keypointrcnn_resnet50_fpn(pretrained=True).to(device).eval()

trf = t.Compose([
    t.ToTensor()
])

with open(csv_file, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow('Video', frame)

        key = cv2.waitKey(1)
        if key in [ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6')]:
            input_img = trf(frame).to(device)
            out = model([input_img])[0]

            keypoints = out['keypoints'][0].detach().cpu().numpy()[:, :2]
            if key == ord('1'):
                label = 'Stand'
            elif key == ord('2'):
                label = 'Sit_chair'
            elif key == ord('3'):
                label = 'Sit_floor'
            elif key == ord('4'):
                label = 'FallingDown'
            elif key == ord('5'):
                label = 'FallDown'


            row = {}
            for i in range(len(keypoints)):
                row[f'keypoint_{i + 1}_x'] = keypoints[i][0]
                row[f'keypoint_{i + 1}_y'] = keypoints[i][1]
            row['label'] = label
            writer.writerow(row)
            print(f"Captured frame with label: {label}")

        elif key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
