import csv
import cv2
import os
from PIL import Image
import torch
from torchvision import models
import torchvision.transforms as t
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.detection.keypointrcnn_resnet50_fpn(pretrained=True).to(device).eval()

trf = t.Compose([
    t.ToTensor()
])

codes = [
    Path.MOVETO,
    Path.LINETO,
    Path.LINETO
]


def make_skeleton(img):
    input_img = trf(img).to(device)
    out = model([input_img])[0]
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    key = torch.zeros((17, 3))
    threshold = 0.9
    label_ = ''
    for box, score, points in zip(out['boxes'], out['scores'], out['keypoints']):
        score = score.detach().cpu().numpy()
        if score < threshold:
            continue
        box = box.detach().cpu().numpy()
        points = points.detach().cpu().numpy()[:, :2]

        rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=2, edgecolor='white',
                                 facecolor='none')
        ax.add_patch(rect)
        for i in range(2):
            path = Path(points[5 + i:10 + i:2], codes)
            line = patches.PathPatch(path, linewidth=2, facecolor='none', edgecolor='red')
            ax.add_patch(line)

        for i in range(2):
            path = Path(points[11 + i:16 + i:2], codes)
            line = patches.PathPatch(path, linewidth=2, facecolor='none', edgecolor='red')
            ax.add_patch(line)

        for i, k in enumerate(points):
            if i < 5:
                radius = 5
                face_color = 'yellow'
            else:
                radius = 10
                face_color = 'red'
            circle = patches.Circle((k[0], k[1]), radius=radius, facecolor=face_color)
            ax.add_patch(circle)

        key = points
    plt.show()
    label = int(input("Label(0 : Sit, 1 : Fall_Down, 2 : Lay_Down, 3: Stumble, 4 : Stand) : "))
    if label == 0:
        label_ = "Sit"
    elif label == 1:
        label_ = "Fall_Down"
    elif label == 2:
        label_ = "Lay_Down"
    elif label == 3:
        label_ = "Stumble"
    elif label == 4:
        label_ = "Stand"
    return key, label_


# CSV 파일을 저장할 경로 및 파일명
csv_file = 'captured_images/pos_data.csv'

# CSV 파일에 헤더를 작성할 때 사용할 필드 이름 생성
fieldnames = []
for i in range(17):
    fieldnames.append(f'keypoint_{i + 1}_x')
    fieldnames.append(f'keypoint_{i + 1}_y')
fieldnames.append('label')

# CSV 파일을 열고 writer 객체를 생성하여 데이터를 작성
with open(csv_file, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()

    directory = 'captured_images'
    file_list = os.listdir(directory)
    print("DATA 갯수:", len(file_list))

    for cnt in range(len(file_list) - 2):
        IMAGE_SIZE = 800
        file_name = f'{directory}/image_{cnt}.jpg'
        img_Data = Image.open(file_name)
        img_Data = img_Data.resize((IMAGE_SIZE, int(img_Data.height * IMAGE_SIZE / img_Data.width)))

        Sceleton_1, label = make_skeleton(img_Data)

        # CSV 파일에 한 줄씩 데이터를 작성
        row = {}
        for i in range(len(Sceleton_1)):
            row[f'keypoint_{i + 1}_x'] = Sceleton_1[i][0]
            row[f'keypoint_{i + 1}_y'] = Sceleton_1[i][1]
        row['label'] = label
        writer.writerow(row)
