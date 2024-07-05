import csv
import cv2
import os
import torch
from torchvision import models
import torchvision.transforms as t

# 비디오 캡쳐 설정
video_path = 'C:/Users/wns20/PycharmProjects/SMART_CCTV/output.avi'
cap = cv2.VideoCapture(video_path)

# CSV 파일 저장 경로 설정
csv_file = 'C:/Users/wns20/PycharmProjects/SMART_CCTV/captured_images/pos_data.csv'

# 필드명 설정
fieldnames = []
for i in range(17):
    fieldnames.append(f'keypoint_{i + 1}_x')
    fieldnames.append(f'keypoint_{i + 1}_y')
fieldnames.append('label')

# 모델 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.detection.keypointrcnn_resnet50_fpn(pretrained=True).to(device).eval()

# 이미지 변환 설정
trf = t.Compose([
    t.ToTensor()
])

# CSV 파일 열기
with open(csv_file, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow('Video', frame)

        key = cv2.waitKey(1)
        if key in [ord('1'), ord('2'), ord('3')]:
            input_img = trf(frame).to(device)
            out = model([input_img])[0]

            keypoints = out['keypoints'][0].detach().cpu().numpy()[:, :2]
            if key == ord('1'):
                label = 'Stand'
            elif key == ord('2'):
                label = 'Sit_chair'
            elif key == ord('3'):
                label = 'Sit_floor'

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
