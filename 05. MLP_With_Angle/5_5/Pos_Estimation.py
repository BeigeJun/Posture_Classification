import cv2
import torch
import numpy as np
import torch.nn as nn
from torchvision import models, transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

keypoint_model = models.detection.keypointrcnn_resnet50_fpn(pretrained=True).to(device).eval()

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers=1):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

class MLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 1024)
        self.fc4 = nn.Linear(1024, 1024)
        self.fc5 = nn.Linear(1024, 256)
        self.fc6 = nn.Linear(256, 128)
        self.fc7 = nn.Linear(128, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = nn.ReLU()(out)
        out = self.fc2(out)
        out = nn.ReLU()(out)
        out = self.fc3(out)
        out = nn.ReLU()(out)
        out = self.fc4(out)
        out = nn.ReLU()(out)
        out = self.fc5(out)
        out = nn.ReLU()(out)
        out = self.fc6(out)
        out = nn.ReLU()(out)
        out = self.fc7(out)
        return out

first_mlp_model = MLP(input_size=12, num_classes=6)
first_mlp_model.load_state_dict(torch.load('/05. MLP_With_Angle/Model/MLP_Remove_Terrified_6Label.pth'))
first_mlp_model = first_mlp_model.to(device).eval()

lstm_model = LSTM(input_size=1, hidden_size=64, num_classes=2).to(device)
lstm_model.load_state_dict(torch.load('lstm_pose_model.pth'))
lstm_model.eval()

def make_angle(point1, point2):
    if point1[0] - point2[0] != 0:
        slope = (point1[1] - point2[1]) / (point1[0] - point2[0])
    else:
        slope = 0
    return slope

First_MLP_label_map = {0: 'FallDown', 1: 'FallingDown', 2: 'Sit_chair', 3: 'Sit_floor', 4: 'Sleep', 5: 'Stand'}
pose_sequence = []

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    transform = transforms.Compose([transforms.ToTensor()])
    input_tensor = transform(frame).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = keypoint_model(input_tensor)

    for output in outputs:
        scores = output['scores'].cpu().numpy()
        high_scores_idx = np.where(scores > 0.95)[0]

        if len(high_scores_idx) > 0:
            keypoints = output['keypoints'][high_scores_idx[0]].cpu().numpy()
            keypoint_scores = output['keypoints_scores'][high_scores_idx[0]].cpu().numpy()
            boxes = output['boxes'][high_scores_idx[0]].cpu().numpy()
            check_count = sum(kp_score < 0.9 for kp_score in keypoint_scores)

            if check_count < 2:
                angles = [
                    make_angle(keypoints[5], keypoints[6]),
                    make_angle(keypoints[5], keypoints[7]),
                    make_angle(keypoints[7], keypoints[9]),
                    make_angle(keypoints[6], keypoints[8]),
                    make_angle(keypoints[8], keypoints[10]),
                    make_angle(keypoints[5], keypoints[11]),
                    make_angle(keypoints[6], keypoints[12]),
                    make_angle(keypoints[11], keypoints[12]),
                    make_angle(keypoints[11], keypoints[13]),
                    make_angle(keypoints[13], keypoints[15]),
                    make_angle(keypoints[12], keypoints[14]),
                    make_angle(keypoints[14], keypoints[16]),
                ]

                angles_tensor = torch.tensor(angles, dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    prediction = first_mlp_model(angles_tensor)
                    _, predicted_label = torch.max(prediction, 1)
                First_Label = First_MLP_label_map[predicted_label.item()]

                pose_sequence.append(predicted_label.item())

                if len(pose_sequence) > 5:
                    pose_sequence.pop(0)

                if len(pose_sequence) == 5:
                    pose_sequence_tensor = torch.tensor(pose_sequence, dtype=torch.float32).view(1, -1, 1).to(device)
                    with torch.no_grad():
                        lstm_prediction = lstm_model(pose_sequence_tensor)
                        _, danger_status = torch.max(lstm_prediction, 1)

                    box_color = (0, 255, 0) if danger_status.item() == 0 else (0, 0, 255)
                    x1, y1, x2, y2 = map(int, boxes)
                    status_text = f"{First_Label} - {'Safe' if danger_status.item() == 0 else 'Danger'}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                    cv2.putText(frame, status_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, box_color, 2)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
