import cv2
import torch
from torchvision import models, transforms
import numpy as np
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

keypoint_model = models.detection.keypointrcnn_resnet50_fpn(pretrained=True).to(device).eval()
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)


def preprocess(image):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0).to(device)

class MLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MLP, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


mlp_model = MLP(input_size=12, num_classes=3)  # 예: input_size=12, num_classes=3
mlp_model.load_state_dict(torch.load('model_MLP.pth'))
mlp_model = mlp_model.to(device).eval()


def make_angle(point1, point2):
    dx = point1[0] - point2[0]
    dy = point1[1] - point2[1]
    if dx != 0:
        slope = dy / dx
    else:
        slope = 0
    return slope


label_map = {0: 'Sit', 1: 'FallDown', 2: 'Stand'}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    input_tensor = preprocess(frame)

    with torch.no_grad():
        outputs = keypoint_model(input_tensor)

    if len(outputs) > 0:
        output = outputs[0]
        scores = output['scores'].cpu().numpy()
        high_scores_idx = np.where(scores > 0.9)[0]

        if len(high_scores_idx) > 0:
            keypoints = output['keypoints'][high_scores_idx[0]].cpu().numpy()
            boxes = output['boxes'][high_scores_idx[0]].cpu().numpy()

            angles = []
            angles.append(make_angle(keypoints[5], keypoints[6]))  # 왼쪽 어깨 -> 오른쪽 어깨
            angles.append(make_angle(keypoints[5], keypoints[7]))  # 왼쪽 어깨 -> 왼쪽 팔꿈치
            angles.append(make_angle(keypoints[7], keypoints[9]))  # 왼쪽 팔꿈치 -> 왼쪽 손목
            angles.append(make_angle(keypoints[6], keypoints[8]))  # 오른쪽 어깨 -> 오른쪽 팔꿈치
            angles.append(make_angle(keypoints[8], keypoints[10])) # 오른쪽 팔꿈치 -> 오른쪽 손목
            angles.append(make_angle(keypoints[5], keypoints[11])) # 왼쪽 어깨 -> 왼쪽 골반
            angles.append(make_angle(keypoints[6], keypoints[12])) # 오른쪽 어깨 -> 오른쪽 골반
            angles.append(make_angle(keypoints[11], keypoints[12]))# 왼쪽 골반 -> 오른쪽 골반
            angles.append(make_angle(keypoints[11], keypoints[13]))# 왼쪽 골반 -> 왼쪽 무릎
            angles.append(make_angle(keypoints[13], keypoints[15]))# 왼쪽 무릎 -> 왼쪽 발목
            angles.append(make_angle(keypoints[12], keypoints[14]))# 오른쪽 골반 -> 오른쪽 무릎
            angles.append(make_angle(keypoints[14], keypoints[16]))# 오른쪽 무릎 -> 오른쪽 발목

            angles_tensor = torch.tensor(angles, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                prediction = mlp_model(angles_tensor)
                _, predicted_label = torch.max(prediction, 1)

            action_label = label_map[predicted_label.item()]

            x1, y1, x2, y2 = map(int, boxes)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            (label_width, label_height), baseline = cv2.getTextSize(action_label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            cv2.rectangle(frame, (x1, y1 - label_height - baseline), (x1 + label_width, y1), (255, 0, 0), cv2.FILLED)

            cv2.putText(frame, action_label, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
