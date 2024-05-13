import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
import cv2
import torchvision.transforms as t
import numpy as np
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Skeleton_Model = models.detection.keypointrcnn_resnet50_fpn(pretrained=True).to(device).eval()

trf = t.Compose([
    t.ToTensor()
])


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def make_skeleton(img):
    img_np = np.array(img)
    img_pil = Image.fromarray(img_np)

    input_img = trf(img_pil).to(device)
    out = Skeleton_Model([input_img])[0]
    key = torch.zeros((17, 2))
    threshold = 0.9
    for box, score, points in zip(out['boxes'], out['scores'], out['keypoints']):
        score = score.detach().cpu().numpy()
        if score < threshold:
            continue
    return key


input_size = 34
hidden_size = 64
output_size = 5
model_path = 'model.pth'
model = MLP(input_size, hidden_size, output_size)
model.load_state_dict(torch.load(model_path))
model.eval()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    IMAGE_SIZE = 800
    img_Data = Image.fromarray(frame)
    img_Data = img_Data.resize((IMAGE_SIZE, int(img_Data.height * IMAGE_SIZE / img_Data.width)))
    key = make_skeleton(img_Data)


    with torch.no_grad():
        input_tensor = torch.tensor(key.flatten(), dtype=torch.float32).unsqueeze(0)
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        posture_label = predicted.item()

    if posture_label == 0:
        posture_text = "Sit"
    elif posture_label == 1:
        posture_text = "Fall Down"
    elif posture_label == 2:
        posture_text = "Lay Down"
    elif posture_label == 3:
        posture_text = "Stumble"
    elif posture_label == 4:
        posture_text = "Stand"

    cv2.putText(frame, f"Posture: {posture_text}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Posture Estimation', frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
