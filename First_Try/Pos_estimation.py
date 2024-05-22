import torch
import torch.nn as nn
from torchvision import models
import cv2
import torchvision.transforms as t
import numpy as np
from PIL import Image
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Skeleton_Model = models.detection.keypointrcnn_resnet50_fpn(pretrained=True).to(device).eval()

trf = t.Compose([
    t.ToTensor()
])

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, output_size):
        super(MLP, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.fc4 = nn.Linear(hidden_size3, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)

        return x

def make_skeleton(img):
    img_np = np.array(img)
    img_pil = Image.fromarray(img_np)

    input_img = trf(img_pil).to(device)
    out = Skeleton_Model([input_img])[0]
    key = torch.zeros((17, 2))
    threshold = 0.9
    for score, points in zip(out['scores'], out['keypoints']):
        if score >= threshold:
            key = points[:, :2].detach().cpu()
            break
    re_key = torch.zeros((8, 2))
    re_key[0][0] = key[5][0]
    re_key[0][1] = key[5][1]
    re_key[1][0] = key[6][0]
    re_key[1][1] = key[6][1]
    for i in range(11, 17):
        re_key[i - 9][0] = key[i][0]
        re_key[i - 9][1] = key[i][1]
    return re_key

def make_center_pos(key):
    np_key = np.array(key)
    max_x = np.max(np_key[:, 0])
    min_x = np.min(np_key[:, 0])
    max_y = np.max(np_key[:, 1])
    min_y = np.min(np_key[:, 1])
    center_x = (max_x + min_x) / 2
    center_y = (max_y + min_y) / 2
    center_pos = [center_x, center_y]
    return center_pos


def remake_pos(keypoints, center_pos):
    new_keypoints = []
    for pos in keypoints:
        new_pos = [pos[0].item() - center_pos[0] + 1000, pos[1].item() - center_pos[1] + 1000]
        new_keypoints.append(new_pos)

    for pos in keypoints:
        pos[0] = (pos[0] - Min) / (Max - Min)
        pos[1] = (pos[1] - Min) / (Max - Min)

    return new_keypoints

input_size = 16
hidden_size1 = 64
hidden_size2 = 128
hidden_size3 = 64
output_size = 2
model_path = 'model.pth'
model = MLP(input_size, hidden_size1, hidden_size2, hidden_size3, output_size).to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

cap = cv2.VideoCapture(0)

with open('C:/Users/wns20/PycharmProjects/SMART_CCTV/First_Try/Max_Min.txt', 'r') as file:
    lines = file.readlines()
Max = float(lines[0].strip())
Min = float(lines[1].strip())

while True:
    ret, frame = cap.read()
    if not ret:
        break

    IMAGE_SIZE = 800
    img_Data = Image.fromarray(frame)
    img_Data = img_Data.resize((IMAGE_SIZE, int(img_Data.height * IMAGE_SIZE / img_Data.width)))
    key = make_skeleton(img_Data)
    if key.numel() == 0:
        continue
    center = make_center_pos(key)
    remake_key = remake_pos(key, center)

    with torch.no_grad():
        input_tensor = torch.tensor(remake_key, dtype=torch.float32).flatten().unsqueeze(0).to(device)
        output = model(input_tensor)
        print(output)
        _, predicted = torch.max(output, 1)
        posture_label = predicted.item()
        print(posture_label)
    posture_text = ''
    if posture_label == 0:
        posture_text = "Sit"
    elif posture_label == 1:
        posture_text = "Stand"

    cv2.putText(frame, f"Posture: {posture_text}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Posture Estimation', frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
