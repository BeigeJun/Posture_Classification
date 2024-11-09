import cv2
import torch
from torchvision import models, transforms
import numpy as np
import torch.nn as nn
from collections import Counter
import tkinter as tk
from tkinter import messagebox
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

keypoint_model = models.detection.keypointrcnn_resnet50_fpn(pretrained=True).to(device).eval()
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)


def show_message():
    messagebox.showinfo("경고", "낙상이 감지 되었습니다!")


def preprocess(image):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0).to(device)


def draw_keypoints_and_connections(background, keypoints):
    cnt = 0
    for point in keypoints:
        if cnt > 5:
            x, y = map(int, point[:2])
            cv2.circle(background, (x, y), 5, (0, 0, 255), -1)
        cnt += 1

    connections = [
        (5, 6), (5, 11), (6, 12), (11, 12),
        (5, 7), (7, 9), (6, 8), (8, 10),
        (11, 13), (13, 15), (12, 14), (14, 16)
    ]

    for connection in connections:
        start_point = tuple(map(int, keypoints[connection[0]][:2]))
        end_point = tuple(map(int, keypoints[connection[1]][:2]))
        cv2.line(background, start_point, end_point, (0, 255, 0), 2)

    head_x = int((keypoints[3, 0] + keypoints[4, 0]) / 2)
    head_y = int((keypoints[3, 1] + keypoints[4, 1]) / 2)
    cv2.circle(background, (head_x, head_y), 10, (0, 0, 255), -1)

    return background



class MLP(nn.Module):
    def __init__(self, input_size, f1_num, f2_num, f3_num, f4_num, f5_num, f6_num, d1, d2, d3, d4, d5, num_classes):
        super(MLP, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_size, f1_num)
        self.fc2 = nn.Linear(f1_num, f2_num)
        self.fc3 = nn.Linear(f2_num, f3_num)
        self.fc4 = nn.Linear(f3_num, f4_num)
        self.fc5 = nn.Linear(f4_num, f5_num)
        self.fc6 = nn.Linear(f5_num, f6_num)
        self.fc7 = nn.Linear(f6_num, num_classes)
        self.dropout1 = nn.Dropout(p=d1)
        self.dropout2 = nn.Dropout(p=d2)
        self.dropout3 = nn.Dropout(p=d3)
        self.dropout4 = nn.Dropout(p=d4)
        self.dropout5 = nn.Dropout(p=d5)

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


model = MLP(12, 64, 128, 256, 256, 128, 64, 0.2, 0.2, 0.2, 0.2, 0.2, 6)
model.load_state_dict(torch.load('Bottom_Loss_Validation_MLP.pth'))
first_mlp_model = model.to(device).eval()
root = tk.Tk()
root.withdraw()


def make_angle(point1, point2):
    if point1[0] - point2[0] != 0:
        slope = (point1[1] - point2[1]) / (point1[0] - point2[0])
    else:
        slope = 0
    return slope


First_MLP_label_map = {0: 'FallDown', 1: 'FallingDown', 2: 'Sit_chair', 3: 'Sit_floor', 4: 'Sleep', 5: 'Stand'}

mode = input(
    "Select mode (1: Single detection, 2: Multiple detection, 3: Secret mode (Single), 4: Secret mode (Multiple)): ")

Label_List = []
nNotDetected = 0
boolHumanCheck = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    input_tensor = preprocess(frame)
    with torch.no_grad():
        outputs = keypoint_model(input_tensor)

    output = outputs[0]
    scores = output['scores'].cpu().numpy()
    high_scores_idx = np.where(scores > 0.95)[0]

    if mode in ['3', '4']:
        display_frame = np.zeros_like(frame)
    else:
        display_frame = frame.copy()

    for idx in high_scores_idx:
        keypoints = output['keypoints'][idx].cpu().numpy()
        keypoint_scores = output['keypoints_scores'][idx].cpu().numpy()
        boxes = output['boxes'][idx].cpu().numpy()

        if mode in ['3', '4']:
            display_frame = draw_keypoints_and_connections(display_frame, keypoints)

        check_count = sum(1 for kp_score in keypoint_scores if kp_score < 0.9)

        if check_count < 2:
            angles = [make_angle(keypoints[i], keypoints[j]) for i, j in [
                (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11),
                (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)
            ]]

            angles_tensor = torch.tensor(angles, dtype=torch.float32).unsqueeze(0).to(device)
            start_time = time.time() * 1000
            with torch.no_grad():
                prediction = first_mlp_model(angles_tensor)
                _, predicted_label = torch.max(prediction, 1)
            end_time = time.time() * 1000
            First_Label = First_MLP_label_map[predicted_label.item()]
            print(f"prediction time : {end_time - start_time:.3f} ms")

            if mode in ['1', '3']:
                if len(Label_List) >= 10:
                    Label_List.pop(0)
                Label_List.append(predicted_label.item())

                if len(Label_List) >= 10:
                    if Label_List[9] == 0:
                        counterBefore = Counter(Label_List[0:11])
                        most_common_count_Before = counterBefore.most_common(1)[0][1]
                        counterBeforeLabel = counterBefore.most_common(1)[0][0]

                        if most_common_count_Before >= 7 and (counterBeforeLabel == 1 or counterBeforeLabel == 4):
                            box_color = (0, 0, 255)
                            show_message()
                            Label_List.clear()
                        elif 1 in Label_List and nNotDetected >= 4:
                            box_color = (0, 0, 255)
                            show_message()
                            Label_List.clear()
                        else:
                            box_color = (0, 255, 100)
                    else:
                        box_color = (0, 255, 0)
                else:
                    box_color = (0, 255, 0)
            else:
                box_color = (0, 255, 0)

            x1, y1, x2, y2 = map(int, boxes)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), box_color, 2)

            (label_width, label_height), baseline = cv2.getTextSize(First_Label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            cv2.rectangle(display_frame, (x1, y1 - label_height - baseline), (x1 + label_width, y1), box_color,
                          cv2.FILLED)
            cv2.putText(display_frame, First_Label, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
                        2)

            if mode in ['1', '3']:
                boolHumanCheck = True
                nNotDetected = 0
            else:
                boolHumanCheck = False
                if nNotDetected < 5:
                    nNotDetected += 1

        if mode in ['1', '3']:
            break

    cv2.imshow('frame', display_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()