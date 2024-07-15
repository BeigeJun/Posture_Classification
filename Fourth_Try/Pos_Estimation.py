import cv2
import torch
from torchvision import models, transforms
import numpy as np
import torch.nn as nn
print(torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

keypoint_model = models.detection.keypointrcnn_resnet50_fpn(pretrained=True).to(device).eval()
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)


def preprocess(image):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0).to(device)

class First_MLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super(First_MLP, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, num_classes)
        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.5)
        self.dropout3 = nn.Dropout(p=0.3)

    def forward(self, x):
        out = self.dropout1(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout1(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout2(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.dropout2(out)
        out = self.fc4(out)
        out = self.relu(out)
        out = self.dropout3(out)
        out = self.fc5(out)
        return out

class Class0_MLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Class0_MLP, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, num_classes)
        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.5)
        self.dropout3 = nn.Dropout(p=0.3)

    def forward(self, x):
        out = self.dropout1(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout1(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout2(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.dropout2(out)
        out = self.fc4(out)
        out = self.relu(out)
        out = self.dropout3(out)
        out = self.fc5(out)
        return out

class Class1_MLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Class1_MLP, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, num_classes)
        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.5)
        self.dropout3 = nn.Dropout(p=0.3)

    def forward(self, x):
        out = self.dropout1(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout1(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout2(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.dropout2(out)
        out = self.fc4(out)
        out = self.relu(out)
        out = self.dropout3(out)
        out = self.fc5(out)
        return out

class Class2_MLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Class2_MLP, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, num_classes)
        self.dropout1 = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout(p=0.5)
        self.dropout3 = nn.Dropout(p=0.3)

    def forward(self, x):
        out = self.dropout1(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout1(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout2(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.dropout2(out)
        out = self.fc4(out)
        out = self.relu(out)
        out = self.dropout3(out)
        out = self.fc5(out)
        return out

first_mlp_model = First_MLP(input_size=12, num_classes=3)
class0_mlp_model = Class0_MLP(input_size=12, num_classes=4)
class1_mlp_model = Class1_MLP(input_size=12, num_classes=2)
class2_mlp_model = Class2_MLP(input_size=12, num_classes=3)
first_mlp_model.load_state_dict(torch.load('C:/Users/wns20/PycharmProjects/SMART_CCTV/Fourth_Try/Model/First_MLP.pth'))
class0_mlp_model.load_state_dict(torch.load('C:/Users/wns20/PycharmProjects/SMART_CCTV/Fourth_Try/Model/Class0Model.pth'))
class1_mlp_model.load_state_dict(torch.load('C:/Users/wns20/PycharmProjects/SMART_CCTV/Fourth_Try/Model/Class1Model.pth'))
class2_mlp_model.load_state_dict(torch.load('C:/Users/wns20/PycharmProjects/SMART_CCTV/Fourth_Try/Model/Class2Model.pth'))
first_mlp_model = first_mlp_model.to(device).eval()
class0_mlp_model = class0_mlp_model.to(device).eval()
class1_mlp_model = class1_mlp_model.to(device).eval()
class2_mlp_model = class2_mlp_model.to(device).eval()


def make_angle(point1, point2):
    dx = point1[0] - point2[0]
    dy = point1[1] - point2[1]
    if dx != 0:
        slope = dy / dx
    else:
        slope = 0
    return slope


First_MLP_label_map = {0: 'Class0', 1: 'Class1', 2: 'Class2'}
Class0_MLP_label_map = {0: 'Guard', 1: 'Kick', 2: 'Punching', 3: 'Stand'}
Class1_MLP_label_map = {0: 'FallingDown', 1: 'Sit_chair'}
Class2_MLP_label_map = {0: 'FallDown', 1: 'Sit_floor', 2: 'Terrified'}
Label_List = []
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
            action_label = ''
            with torch.no_grad():
                prediction = first_mlp_model(angles_tensor)
                _, predicted_label = torch.max(prediction, 1)
            action_label_First = First_MLP_label_map[predicted_label.item()]

            if action_label_First == First_MLP_label_map[0]:
                with torch.no_grad():
                    prediction = class0_mlp_model(angles_tensor)
                    _, predicted_label = torch.max(prediction, 1)
                action_label = Class0_MLP_label_map[predicted_label.item()]

            elif action_label_First == First_MLP_label_map[1]:
                with torch.no_grad():
                    prediction = class1_mlp_model(angles_tensor)
                    _, predicted_label = torch.max(prediction, 1)
                action_label = Class1_MLP_label_map[predicted_label.item()]

            elif action_label_First == First_MLP_label_map[2]:
                with torch.no_grad():
                    prediction = class2_mlp_model(angles_tensor)
                    _, predicted_label = torch.max(prediction, 1)
                action_label = Class2_MLP_label_map[predicted_label.item()]

            x1, y1, x2, y2 = map(int, boxes)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            Out_Label = action_label_First + '_' + action_label

            (label_width, label_height), baseline = cv2.getTextSize(Out_Label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            cv2.rectangle(frame, (x1, y1 - label_height - baseline), (x1 + label_width, y1), (255, 0, 0), cv2.FILLED)
            cv2.putText(frame, Out_Label, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
