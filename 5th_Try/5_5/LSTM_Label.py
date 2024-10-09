import cv2
import torch
from torchvision import models, transforms
import numpy as np
import torch.nn as nn
import csv
import os

# GPU 사용 가능 여부 확인
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 사전 학습된 키포인트 모델 로드
keypoint_model = models.detection.keypointrcnn_resnet50_fpn(pretrained=True).to(device).eval()

# 이미지 전처리 함수
def preprocess(image):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0).to(device)

# MLP 모델 정의
class MLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MLP, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 1024)
        self.fc4 = nn.Linear(1024, 1024)
        self.fc5 = nn.Linear(1024, 256)
        self.fc6 = nn.Linear(256, 128)
        self.fc7 = nn.Linear(128, num_classes)
        self.dropout1 = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout(p=0.3)
        self.dropout3 = nn.Dropout(p=0.4)
        self.dropout4 = nn.Dropout(p=0.5)
        self.dropout5 = nn.Dropout(p=0.5)

    def forward(self, x):
        out = self.dropout1(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout2(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout3(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.dropout4(out)
        out = self.fc4(out)
        out = self.relu(out)
        out = self.dropout3(out)
        out = self.fc5(out)
        out = self.relu(out)
        out = self.dropout2(out)
        out = self.fc6(out)
        out = self.relu(out)
        out = self.dropout1(out)
        out = self.fc7(out)
        return out

# MLP 모델 로드
first_mlp_model = MLP(input_size=12, num_classes=6)
first_mlp_model.load_state_dict(
    torch.load('C:/Users/wns20/PycharmProjects/SMART_CCTV/5th_Try/Model/MLP_Remove_Terrified_6Label.pth'))
first_mlp_model = first_mlp_model.to(device).eval()

# 각 포인트 사이 각도 계산 함수
def make_angle(point1, point2):
    if point1[0] - point2[0] != 0:
        slope = (point1[1] - point2[1]) / (point1[0] - point2[0])
    else:
        slope = 0
    return slope

# MLP 예측을 위한 레이블 맵
First_MLP_label_map = {0: 'FallDown', 1: 'FallingDown', 2: 'Sit_chair', 3: 'Sit_floor', 4: 'Sleep', 5: 'Stand'}

# 포즈 레이블을 저장할 CSV 파일
output_csv = 'pose_labels.csv'

# CSV 파일 초기화
with open(output_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Video", "Labels", "Final_Label"])  # 비디오 파일명, 레이블들, 최종 레이블

# 비디오가 들어 있는 폴더 경로
video_folder = 'C:/Users/wns20/PycharmProjects/SMART_CCTV/5th_Try/5_5/TestVideo/Fall'

# 폴더 내 비디오 파일 처리
for video_file in os.listdir(video_folder):
    if video_file.endswith('.mp4'):
        video_path = os.path.join(video_folder, video_file)
        cap = cv2.VideoCapture(video_path)

        frame_count = 0
        Label_List = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            input_tensor = preprocess(frame)
            with torch.no_grad():
                outputs = keypoint_model(input_tensor)

            for i in range(len(outputs)):
                output = outputs[i]
                scores = output['scores'].cpu().numpy()
                high_scores_idx = np.where(scores > 0.95)[0]

                if len(high_scores_idx) > 0:
                    keypoints = output['keypoints'][high_scores_idx[0]].cpu().numpy()
                    keypoint_scores = output['keypoints_scores'][high_scores_idx[0]].cpu().numpy()
                    check_count = sum([1 for kp_score in keypoint_scores if kp_score < 0.9])

                    if check_count < 2:
                        angles = []
                        angles.append(make_angle(keypoints[5], keypoints[6]))  # 왼쪽 어깨 -> 오른쪽 어깨
                        angles.append(make_angle(keypoints[5], keypoints[7]))  # 왼쪽 어깨 -> 왼쪽 팔꿈치
                        angles.append(make_angle(keypoints[7], keypoints[9]))  # 왼쪽 팔꿈치 -> 왼쪽 손목
                        angles.append(make_angle(keypoints[6], keypoints[8]))  # 오른쪽 어깨 -> 오른쪽 팔꿈치
                        angles.append(make_angle(keypoints[8], keypoints[10]))  # 오른쪽 팔꿈치 -> 오른쪽 손목
                        angles.append(make_angle(keypoints[5], keypoints[11]))  # 왼쪽 어깨 -> 왼쪽 골반
                        angles.append(make_angle(keypoints[6], keypoints[12]))  # 오른쪽 어깨 -> 오른쪽 골반
                        angles.append(make_angle(keypoints[11], keypoints[12]))  # 왼쪽 골반 -> 오른쪽 골반
                        angles.append(make_angle(keypoints[11], keypoints[13]))  # 왼쪽 골반 -> 왼쪽 무릎
                        angles.append(make_angle(keypoints[13], keypoints[15]))  # 왼쪽 무릎 -> 왼쪽 발목
                        angles.append(make_angle(keypoints[12], keypoints[14]))  # 오른쪽 골반 -> 오른쪽 무릎
                        angles.append(make_angle(keypoints[14], keypoints[16]))  # 오른쪽 무릎 -> 오른쪽 발목

                        angles_tensor = torch.tensor(angles, dtype=torch.float32).unsqueeze(0).to(device)
                        with torch.no_grad():
                            prediction = first_mlp_model(angles_tensor)
                            _, predicted_label = torch.max(prediction, 1)

                        First_Label = First_MLP_label_map[predicted_label.item()]

                        # 레이블을 Label_List에 추가
                        if len(Label_List) >= 20:
                            Label_List.pop(0)
                        Label_List.append(predicted_label.item())

            frame_count += 1

        # 비디오 처리 후 레이블을 CSV에 저장
        final_labels = ','.join(map(str, Label_List))
        with open(output_csv, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([video_file, final_labels, "Danger"])  # 비디오 파일명, 레이블들, 최종 레이블 "Danger"

        cap.release()

cv2.destroyAllWindows()
