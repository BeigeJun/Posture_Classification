import torch
import torch.nn as nn
from torchvision import models
import cv2
import torchvision.transforms as t
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Skeleton_Model = models.detection.keypointrcnn_resnet50_fpn(pretrained=True).to(device).eval()

trf = t.Compose([
    t.ToTensor()
])
transform = t.Compose([
    t.ToTensor(),
    t.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

codes = [
    Path.MOVETO,
    Path.LINETO,
    Path.LINETO
]

def _make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return int(new_v)

class h_swish(nn.Module):
    def __init__(self):
        super(h_swish, self).__init__()
        self.relu6 = nn.ReLU6()

    def forward(self, x):
        return x * (self.relu6(x + 3) / 6)

class inverted_residual_block(nn.Module):
    def __init__(self, i, t, o, k, s, re=False, se=False):
        super(inverted_residual_block, self).__init__()
        expansion = int(i * t)
        if re:
            nonlinear = nn.ReLU6()
        else:
            nonlinear = h_swish()
        self.se = se
        self.conv = nn.Sequential(
            nn.Conv2d(i, expansion, 1, 1),
            nn.BatchNorm2d(expansion),
            nonlinear
        )
        self.dconv = nn.Sequential(
            nn.Conv2d(expansion, expansion, k, s, k // 2, groups=expansion),
            nn.BatchNorm2d(expansion),
            nonlinear
        )
        self.semodule = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(expansion, _make_divisible(expansion // 4), 1, 1),
            nn.ReLU(),
            nn.Conv2d(_make_divisible(expansion // 4), expansion, 1, 1),
            h_swish()
        )
        self.linearconv = nn.Sequential(
            nn.Conv2d(expansion, o, 1, 1),
            nn.BatchNorm2d(o)
        )
        self.shortcut = (i == o and s == 1)

    def forward(self, x):
        out = self.conv(x)
        out = self.dconv(out)
        if self.se:
            out *= self.semodule(out)
        out = self.linearconv(out)
        if self.shortcut:
            out += x
        return out

class mobilenetv3(nn.Module):
    def __init__(self, ver=0, w=1.0):
        super(mobilenetv3, self).__init__()
        large = [
            [1, 16, 3, 1, False, False],
            [4, 24, 3, 2, False, False],
            [3, 24, 3, 1, False, False],
            [3, 40, 5, 2, False, True],
            [3, 40, 5, 1, False, True],
            [3, 40, 5, 1, False, True],
            [6, 80, 3, 2, True, False],
            [2.5, 80, 3, 1, True, False],
            [2.4, 80, 3, 1, True, False],
            [2.4, 80, 3, 1, True, False],
            [6, 112, 3, 1, True, True],
            [6, 112, 3, 1, True, True],
            [6, 160, 5, 2, True, True],
            [6, 160, 5, 1, True, True],
            [6, 160, 5, 1, True, True]
        ]

        small = [
            [1, 16, 3, 2, False, True],
            [4, 24, 3, 2, False, False],
            [11.0 / 3.0, 24, 3, 1, False, False],
            [4, 40, 5, 2, True, True],
            [6, 40, 5, 1, True, True],
            [6, 40, 5, 1, True, True],
            [3, 48, 5, 1, True, True],
            [3, 48, 5, 1, True, True],
            [6, 96, 5, 2, True, True],
            [6, 96, 5, 1, True, True],
            [6, 96, 5, 1, True, True],
        ]

        in_channels = _make_divisible(16 * w)

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, in_channels, 3, 2, 1),
            nn.BatchNorm2d(int(16 * w)),
            nn.ReLU6()
        )
        if ver == 0:
            stack = large
            last = 1280
        else:
            stack = small
            last = 1024
        layers = []

        for i in range(len(stack)):
            out_channels = _make_divisible(stack[i][1] * w)
            layers.append(
                inverted_residual_block(in_channels, stack[i][0], out_channels, stack[i][2], stack[i][3], stack[i][4],
                                        stack[i][5]))
            in_channels = out_channels
        self.stack = nn.Sequential(*layers)
        self.last = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * 6, 1, 1),
            nn.BatchNorm2d(out_channels * 6),
            h_swish(),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels * 6, last, 1, 1),
            h_swish(),
            nn.Conv2d(last, 1000, 1, 1)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.stack(out)
        out = self.last(out)
        out = out.view(out.size(0), -1)
        return out

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

    return new_keypoints

def make_img(keypoints):
    center = make_center_pos(keypoints)
    changed_pos = remake_pos(keypoints, center)
    changed_keypoints = np.array(changed_pos)

    plt.figure(figsize=(3, 5))
    plt.scatter(changed_keypoints[5:17, 0], changed_keypoints[5:17, 1], color='red', s=300)
    head_x = (changed_keypoints[3, 0] + changed_keypoints[4, 0]) / 2
    head_y = (changed_keypoints[3, 1] + changed_keypoints[4, 1]) / 2
    plt.scatter(head_x, head_y, color='red', s=500)

    ax = plt.gca()
    ax.axis('off')
    ax.invert_xaxis()
    ax.invert_yaxis()

    # Drawing the skeleton lines
    for i, j in [(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 7), (6, 8), (7, 9), (8, 10), (9, 11), (10, 12), (11, 13)]:
        if changed_keypoints[i][0] > 0 and changed_keypoints[j][0] > 0:
            plt.plot([changed_keypoints[i][0], changed_keypoints[j][0]],
                     [changed_keypoints[i][1], changed_keypoints[j][1]], color='blue', linewidth=2)

    plt.tight_layout()
    plt.savefig(os.path.join('C:/Users/wns20/PycharmProjects/SMART_CCTV/Second_Try', 'Live_img.png'))
    plt.close()

model_path = 'model_google.pth'
model = mobilenetv3().to(device)
model.eval()

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

with open('C:/Users/wns20/PycharmProjects/SMART_CCTV/Second_Try/Max_Min.txt', 'r') as file:
    lines = file.readlines()
Max = float(lines[0].strip())
Min = float(lines[1].strip())
print(Max)
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
        print(predicted)

    # Updated labels based on the six posture categories
    posture_text = ''
    if posture_label == 0:
        posture_text = "Standing"
    elif posture_label == 1:
        posture_text = "Fallen"
    elif posture_label == 2:
        posture_text = "Falling"
    elif posture_label == 3:
        posture_text = "Sitting on Chair"
    elif posture_label == 4:
        posture_text = "Sitting on Floor"
    elif posture_label == 5:
        posture_text = "Panic"
    print(posture_text)
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        (w, h), _ = cv2.getTextSize(posture_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        cv2.rectangle(frame, (x1, y1 - 30), (x1 + w, y1), (255, 0, 0), -1)
        cv2.putText(frame, posture_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.putText(frame, f"Posture: {posture_text}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Posture Estimation', frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
