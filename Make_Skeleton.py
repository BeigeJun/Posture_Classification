from PIL import Image
import torch
from torchvision import models
import torchvision.transforms as t
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import os

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


def make_sceleton(img):
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
            path = Path(points[5+i:10+i:2], codes)
            line = patches.PathPatch(path, linewidth=2, facecolor='none', edgecolor='red')
            ax.add_patch(line)

        for i in range(2):
            path = Path(points[11+i:16+i:2], codes)
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
        label = int(input("Label : "))
        if label == 0:
            label_ = "STAND"
        else:
            label_ = "SIT"
    plt.show()
    return key, label_


directory = 'C:/Users/wns20/PycharmProjects/SMART_CCTV/DATASET' + '/Sit'
file_list = os.listdir(directory)
print("DATA 갯수:", len(file_list))

for cnt in range(len(file_list)-1):
    IMAGE_SIZE = 800
    file_name = f'{directory}/img{cnt}.jpg'
    img_Data = Image.open(file_name)
    img_Data = img_Data.resize((IMAGE_SIZE, int(img_Data.height * IMAGE_SIZE / img_Data.width)))

    Sceleton_1, label = make_sceleton(img_Data)

    file_name = f'{directory}/Pos.txt'
    with open(file_name, 'r') as f:
        Lines = f.read()

    with open(file_name, 'w') as f:
        for i in range(len(Sceleton_1)):
            for j in range(2):
                Lines += str(Sceleton_1[i][j]) + ' '
        Lines += '\n'
        f.write(Lines)
