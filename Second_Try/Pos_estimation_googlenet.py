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
import argparse
import numpy as np
from torch import Tensor
from typing import Optional
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Skeleton_Model = models.detection.keypointrcnn_resnet50_fpn(pretrained=True).to(device).eval()

trf = t.Compose([
    t.ToTensor()
])
transform = transforms.Compose([
    transforms.Resize((300, 500)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

codes = [
    Path.MOVETO,
    Path.LINETO,
    Path.LINETO
]

class Inception(nn.Module):
    def __init__(self, in_channels, n1x1, n3x3_reduce, n3x3, n5x5_reduce, n5x5, pool_proj) -> None:
        super(Inception, self).__init__()
        self.branch1 = ConvBlock(in_channels, n1x1, kernel_size=1, stride=1, padding=0)

        self.branch2 = nn.Sequential(
            ConvBlock(in_channels, n3x3_reduce, kernel_size=1, stride=1, padding=0),
            ConvBlock(n3x3_reduce, n3x3, kernel_size=3, stride=1, padding=1))

        self.branch3 = nn.Sequential(
            ConvBlock(in_channels, n5x5_reduce, kernel_size=1, stride=1, padding=0),
            ConvBlock(n5x5_reduce, n5x5, kernel_size=5, stride=1, padding=2))

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels, pool_proj, kernel_size=1, stride=1, padding=0))

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)
        return torch.cat([x1, x2, x3, x4], dim=1)


class GoogLeNet(nn.Module):
    def __init__(self, aux_logits=True, num_classes=3) -> None:
        super(GoogLeNet, self).__init__()
        assert aux_logits == True or aux_logits == False
        self.aux_logits = aux_logits

        self.conv1 = ConvBlock(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)
        self.conv2 = ConvBlock(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv3 = ConvBlock(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.a3 = Inception(192, 64, 96, 128, 16, 32, 32)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)
        self.a4 = Inception(480, 192, 96, 208, 16, 48, 64)
        self.b4 = Inception(512, 160, 112, 224, 24, 64, 64)
        self.c4 = Inception(512, 128, 128, 256, 24, 64, 64)
        self.d4 = Inception(512, 112, 144, 288, 32, 64, 64)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.4)
        self.linear = nn.Linear(1024, num_classes)

        if self.aux_logits:
            self.aux1 = InceptionAux(512, num_classes)
            self.aux2 = InceptionAux(528, num_classes)
        else:
            self.aux1 = None
            self.aux2 = None

    def transform_input(self, x: Tensor) -> Tensor:
        x_R = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
        x_G = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
        x_B = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        x = torch.cat([x_R, x_G, x_B], 1)
        return x

    def forward(self, x: Tensor) -> Tensor:
        x = self.transform_input(x)

        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool2(x)
        x = self.a3(x)
        x = self.b3(x)
        x = self.maxpool3(x)
        x = self.a4(x)
        aux1: Optional[Tensor] = None
        if self.aux_logits and self.training:
            aux1 = self.aux1(x)

        x = self.b4(x)
        x = self.c4(x)
        x = self.d4(x)
        aux2: Optional[Tensor] = None
        if self.aux_logits and self.training:
            aux2 = self.aux2(x)

        x = self.e4(x)
        x = self.maxpool4(x)
        x = self.a5(x)
        x = self.b5(x)
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)  # x = x.reshape(x.shape[0], -1)
        x = self.linear(x)
        x = self.dropout(x)

        if self.aux_logits and self.training:
            return aux1, aux2
        else:
            return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs) -> None:
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        return x


class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes) -> None:
        super(InceptionAux, self).__init__()
        self.avgpool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = ConvBlock(in_channels, 128, kernel_size=1, stride=1, padding=0)
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
        self.dropout = nn.Dropout(p=0.7)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        x = self.avgpool(x)
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def make_skeleton(img):
    img_np = np.array(img)
    img_pil = Image.fromarray(img_np)

    input_img = trf(img_pil).to(device)
    out = Skeleton_Model([input_img])[0]
    key = torch.zeros((17, 2))
    bbox = None
    threshold = 0.9
    for score, points, box in zip(out['scores'], out['keypoints'], out['boxes']):
        if score >= threshold:
            key = points[:, :2].detach().cpu()
            bbox = box.detach().cpu().numpy()
            break
    return key, bbox

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

    plt.scatter(changed_keypoints[5:17, 0], changed_keypoints[5:17, 1], color='red', s=300)
    head_x = (changed_keypoints[3, 0] + changed_keypoints[4, 0]) / 2
    head_y = (changed_keypoints[3, 1] + changed_keypoints[4, 1]) / 2
    plt.scatter(head_x, head_y, color='red', s=500)

    ax = plt.gca()
    ax.axis('off')
    ax.invert_xaxis()
    ax.invert_yaxis()

    shoulder_verts = changed_keypoints[[5, 6], :2]
    shoulder_path = Path(shoulder_verts, [Path.MOVETO, Path.LINETO])
    shoulder_line = patches.PathPatch(shoulder_path, linewidth=8, facecolor='none', edgecolor='red')
    ax.add_patch(shoulder_line)

    for j in range(2):
        body_verts = changed_keypoints[[5 + j, 11 + j], :2]
        body_path = Path(body_verts, [Path.MOVETO, Path.LINETO])
        body_line = patches.PathPatch(body_path, linewidth=8, facecolor='none', edgecolor='red')
        ax.add_patch(body_line)

    pelvis_verts = changed_keypoints[[11, 12], :2]
    pelvis_path = Path(pelvis_verts, [Path.MOVETO, Path.LINETO])
    pelvis_line = patches.PathPatch(pelvis_path, linewidth=8, facecolor='none', edgecolor='red')
    ax.add_patch(pelvis_line)

    for j in range(2):
        verts = changed_keypoints[[5 + j, 7 + j, 9 + j], :2]
        path = Path(verts, codes)
        line = patches.PathPatch(path, linewidth=2, facecolor='none', edgecolor='red')
        ax.add_patch(line)

    for j in range(2):
        verts = changed_keypoints[[11 + j, 13 + j, 15 + j], :2]
        path = Path(verts, codes)
        line = patches.PathPatch(path, linewidth=8, facecolor='none', edgecolor='red')
        ax.add_patch(line)

    plt.tight_layout()
    plt.savefig(os.path.join('C:/Users/wns20/PycharmProjects/SMART_CCTV/Second_Try', f'Live_img.png'))
    plt.close()

model_path = 'model_google.pth'
model = GoogLeNet().to(device)
model.eval()

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)


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
    key, bbox = make_skeleton(img_Data)
    make_img(key)
    if key.numel() == 0:
        continue

    with torch.no_grad():
        img_path = os.path.join('C:/Users/wns20/PycharmProjects/SMART_CCTV/Second_Try', 'Live_img.png')
        img = Image.open(img_path).convert('RGB')
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
    elif posture_label == 2:
        posture_text = "Falldown"
    print(output)

    if bbox is not None:
        x1, y1, x2, y2 = bbox
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        # Draw text background
        (w, h), _ = cv2.getTextSize(posture_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        cv2.rectangle(frame, (x1, y1 - 30), (x1 + w, y1), (255, 0, 0), -1)
        # Draw text
        cv2.putText(frame, posture_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.putText(frame, f"Posture: {posture_text}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Posture Estimation', frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
