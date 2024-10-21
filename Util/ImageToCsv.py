import os
import csv
from PIL import Image
import torch
from torchvision import models
import torchvision.transforms as t

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.detection.keypointrcnn_resnet50_fpn(pretrained=True).to(device).eval()

# CSV 파일 필드명 설정
fieldnames = []
for i in range(17):
    fieldnames.append(f'keypoint_{i + 1}_x')
    fieldnames.append(f'keypoint_{i + 1}_y')
fieldnames.append('label')

# CSV 파일 저장 경로
csv_file = '/Users/wns20/PycharmProjects/SMART_CCTV/Second_Try/Frames/keypoints_labels.csv'

trf = t.Compose([t.ToTensor()])

def make_skeleton(img):
    input_img = trf(img).to(device)
    out = model([input_img])[0]
    key = torch.zeros((17, 2))  # keypoint는 (x, y) 좌표만 저장
    threshold = 0.999

    for box, score, points in zip(out['boxes'], out['scores'], out['keypoints']):
        score = score.detach().cpu().numpy()
        if score < threshold:
            continue
        points = points.detach().cpu().numpy()[:, :2]

        key = points  # 키포인트 값 저장

    return key  # 키포인트만 반환

# CSV 파일 쓰기 준비
with open(csv_file, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()

    # 루트 디렉토리
    root_dir = '/Users/wns20/PycharmProjects/SMART_CCTV/Second_Try/Frames'

    # 라벨별 디렉토리 순회
    for label in os.listdir(root_dir):
        label_dir = os.path.join(root_dir, label)

        if not os.path.isdir(label_dir):
            continue  # 폴더가 아닌 경우 건너뛰기

        # 해당 라벨 디렉토리의 이미지 파일 순회
        for img_file in os.listdir(label_dir):
            img_path = os.path.join(label_dir, img_file)

            if not img_file.endswith('.jpg'):
                continue  # 이미지 파일이 아닌 경우 건너뛰기

            # 이미지 열기 및 리사이즈
            IMAGE_SIZE = 800
            img_Data = Image.open(img_path)
            img_Data = img_Data.resize((IMAGE_SIZE, int(img_Data.height * IMAGE_SIZE / img_Data.width)))

            # 스켈레톤 추출
            Sceleton_1 = make_skeleton(img_Data)

            if torch.all(torch.tensor(Sceleton_1) == 0):  # Check if all keypoints are zero
                print(f"Skipping {img_file} due to all zero keypoints")
                continue  # Skip if all keypoints are zero

            # 스켈레톤 데이터와 폴더명을 라벨로 CSV 파일에 기록할 행 생성
            row = {}
            for i in range(len(Sceleton_1)):
                row[f'keypoint_{i + 1}_x'] = Sceleton_1[i][0]
                row[f'keypoint_{i + 1}_y'] = Sceleton_1[i][1]
            row['label'] = label  # 해당 이미지의 폴더명을 라벨로 사용

            # 행을 CSV 파일에 쓰기
            writer.writerow(row)

print(f"CSV 파일로 저장 완료: {csv_file}")
