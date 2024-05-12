import numpy as np
from PIL import Image
import torch
from torchvision import models
import torchvision.transforms as t
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.detection.keypointrcnn_resnet50_fpn(pretrained=True).to(device).eval()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow('Pose Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('c'):
        frame = cv2.flip(frame, 1)
        cv2.imwrite('front.jpg', frame)
        break
cap.release()
cv2.destroyAllWindows()

IMAGE_SIZE = 800
img_front = Image.open('front.jpg')
img_front = img_front.resize((IMAGE_SIZE, int(img_front.height * IMAGE_SIZE / img_front.width)))
image_width, image_height = img_front.size

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
    plt.show()
    return key


Sceleton_1 = make_sceleton(img_front)