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
import cv2

cap = cv2.VideoCapture(0)
video_path = 'C:/Users/wns20/PycharmProjects/SMART_CCTV/output.avi'

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(video_path, fourcc, 20.0, (640, 480))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow('Video', frame)

    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()


if not os.path.isfile(video_path):
    print(f'Error: {video_path} not found')
    exit()

cap = cv2.VideoCapture(video_path)

save_directory = '/Users/wns20/PycharmProjects/SMART_CCTV/captured_images'
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow('Video', frame)

    key = cv2.waitKey(1)
    if key == ord('c'):
        file_name = os.path.join(save_directory, f'image_{count}.jpg')
        cv2.imwrite(file_name, frame)
        print(f'Captured image {file_name}')
        count += 1
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



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
    label = int(input("Label(0 : Sit, 9 : Stand) : "))
    if label == 0:
        label_ = "Sit"
    elif label == 9:
        label_ = "Stand"
    return key, label_


csv_file = '/Users/wns20/PycharmProjects/SMART_CCTV/captured_images/pos_data.csv'

fieldnames = []
for i in range(17):
    fieldnames.append(f'keypoint_{i + 1}_x')
    fieldnames.append(f'keypoint_{i + 1}_y')
fieldnames.append('label')

with open(csv_file, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()

    directory = '/Users/wns20/PycharmProjects/SMART_CCTV/captured_images'
    file_list = os.listdir(directory)
    print("DATA 갯수:", len(file_list))

    for cnt in range(len(file_list) - 2):
        IMAGE_SIZE = 800
        file_name = f'{directory}/image_{cnt}.jpg'
        img_Data = Image.open(file_name)
        img_Data = img_Data.resize((IMAGE_SIZE, int(img_Data.height * IMAGE_SIZE / img_Data.width)))

        Sceleton_1, label = make_skeleton(img_Data)

        row = {}
        for i in range(len(Sceleton_1)):
            row[f'keypoint_{i + 1}_x'] = Sceleton_1[i][0]
            row[f'keypoint_{i + 1}_y'] = Sceleton_1[i][1]
        row['label'] = label
        writer.writerow(row)
