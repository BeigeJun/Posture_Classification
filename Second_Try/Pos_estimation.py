import torch
import torch.nn as nn
from torchvision import models
import cv2
import torchvision.transforms as t
from torchvision import datasets, transforms
import numpy as np
from PIL import Image
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Skeleton_Model = models.detection.keypointrcnn_resnet50_fpn(pretrained=True).to(device).eval()

trf = t.Compose([
    t.ToTensor()
])
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
codes = [
    Path.MOVETO,
    Path.LINETO,
    Path.LINETO
]
input_size = 1
hidden_size1 = 9
hidden_size2 = 18
hidden_size3 = 27
hidden_to_lin = 36
lin_to_lin = 100
output_size = 2

class CNN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, hidden_to_lin, lin_to_lin, output_size):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_size, out_channels=hidden_size1, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=hidden_size1, out_channels=hidden_size2, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=hidden_size2, out_channels=hidden_size3, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=hidden_size3, out_channels=hidden_to_lin, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(hidden_to_lin, lin_to_lin)
        self.fc2 = nn.Linear(lin_to_lin, output_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.fc1(x.view(-1, 36)))
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
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
    return key

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

    for i, pos in enumerate(keypoints):
        new_keypoints[i][0] = (pos[0].item() - Min) / (Max - Min)
        new_keypoints[i][1] = (pos[1].item() - Min) / (Max - Min)
    #     print("포스: ", pos)
    #
    # print(new_keypoints)
    return new_keypoints

def make_img(keypoints):
    center = make_center_pos(keypoints)
    changed_pos = remake_pos(keypoints, center)
    changed_keypoints = np.array(changed_pos)
    # print(changed_keypoints)
    plt.figure(figsize=(3, 5))

    plt.scatter(changed_keypoints[5:17, 0], changed_keypoints[5:17, 1], color='red')
    head_x = (changed_keypoints[3, 0] + changed_keypoints[4, 0]) / 2
    head_y = (changed_keypoints[3, 1] + changed_keypoints[4, 1]) / 2
    plt.scatter(head_x, head_y, color='red', s=500)

    ax = plt.gca()
    ax.axis('off')
    ax.invert_xaxis()
    ax.invert_yaxis()

    shoulder_verts = changed_keypoints[[5, 6], :2]
    shoulder_path = Path(shoulder_verts, [Path.MOVETO, Path.LINETO])
    shoulder_line = patches.PathPatch(shoulder_path, linewidth=2, facecolor='none', edgecolor='red')
    ax.add_patch(shoulder_line)

    for j in range(2):
        body_verts = changed_keypoints[[5 + j, 11 + j], :2]
        body_path = Path(body_verts, [Path.MOVETO, Path.LINETO])
        body_line = patches.PathPatch(body_path, linewidth=2, facecolor='none', edgecolor='red')
        ax.add_patch(body_line)

    pelvis_verts = changed_keypoints[[11, 12], :2]
    pelvis_path = Path(pelvis_verts, [Path.MOVETO, Path.LINETO])
    pelvis_line = patches.PathPatch(pelvis_path, linewidth=2, facecolor='none', edgecolor='red')
    ax.add_patch(pelvis_line)

    for j in range(2):
        verts = changed_keypoints[[5 + j, 7 + j, 9 + j], :2]
        path = Path(verts, codes)
        line = patches.PathPatch(path, linewidth=2, facecolor='none', edgecolor='red')
        ax.add_patch(line)

    for j in range(2):
        verts = changed_keypoints[[11 + j, 13 + j, 15 + j], :2]
        path = Path(verts, codes)
        line = patches.PathPatch(path, linewidth=2, facecolor='none', edgecolor='red')
        ax.add_patch(line)

    plt.tight_layout()
    plt.savefig(os.path.join('C:/Users/wns20/PycharmProjects/SMART_CCTV/Second_Try', f'Live_img.png'))
    plt.close()

model_path = 'model.pth'
model = CNN(input_size, hidden_size1, hidden_size2, hidden_size3, hidden_to_lin, lin_to_lin, output_size).to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

cap = cv2.VideoCapture(0)

with open('C:/Users/wns20/PycharmProjects/SMART_CCTV/Second_Try/Max_Min.txt', 'r') as file:
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
    make_img(key)
    if key.numel() == 0:
        continue

    with torch.no_grad():
        img_path = os.path.join('C:/Users/wns20/PycharmProjects/SMART_CCTV/Second_Try', 'Live_img.png')
        img = Image.open(img_path).convert('L')
        img = transform(img).unsqueeze(0).to(device)
        output = model(img)
        _, predicted = torch.max(output, 1)
        posture_label = predicted.item()

        # print(posture_label)
    posture_text = ''
    if posture_label == 0:
        posture_text = "Sit"
    elif posture_label == 1:
        posture_text = "Stand"
    print(output)
    cv2.putText(frame, f"Posture: {posture_text}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Posture Estimation', frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
