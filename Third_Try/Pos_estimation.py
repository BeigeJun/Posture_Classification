import cv2
import torch
from torchvision import models, transforms
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.detection.keypointrcnn_resnet50_fpn(pretrained=True).to(device).eval()

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

def preprocess(image):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0).to(device)


skeleton = [
    [0, 1], [0, 2], [1, 3], [2, 4], # 얼굴
    [0, 5], [0, 6],# 목
    [5, 7], [7, 9], [6, 8], [8, 10], # 팔 5,7,9왼쪽-> 출력 기준 오른쪽 6,8,10 오른쪽 -> 출력 기준 왼쪽
    [5, 6], [5, 11], [6, 12], [11, 12], # 몸통
    [11, 13], [13, 15], [12, 14], [14, 16] # 다리
]


def euclidean_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** (1/2)
    return distance


def angle_between_lines(m1, m2):
    if m1 == m2:
        return 0.0
    if m1 * m2 == -1:
        return 90.0
    try:
        tan_theta = np.abs((m1 - m2) / (1 + m1 * m2))
        theta = np.arctan(tan_theta)
        theta_degrees = np.degrees(theta)
        return theta_degrees
    except ZeroDivisionError:
        return 90.0


def draw_keypoints(image, keypoints, threshold=0.9):
    slopes = []
    Length = []
    for xy_conf_points in keypoints:
        points = []
        for _, (x, y, conf) in enumerate(xy_conf_points):
            if conf > threshold:
                points.append((int(x), int(y)))
                cv2.circle(image, (int(x), int(y)), 5, (0, 255, 0), -1)
            else:
                points.append(None)

        for i, j in skeleton:
            if points[i] and points[j]:
                cv2.line(image, points[i], points[j], (255, 0, 0), 2)
                dx = points[j][0] - points[i][0]
                dy = points[j][1] - points[i][1]
                if dx != 0:
                    slope = dy / dx
                else:
                    slope = float('inf')
                slopes.append(slope)
        Length.append(euclidean_distance(points[5], points[6]))
        Length.append(euclidean_distance(points[11], points[12]))
        #어깨랑 골반 길이로 거리 계산 허나 옆으로 돌아서 짧아지는건 어떻게 해결할지?
        #애초에 캠 하나로 이 문제를 해결할수 있는가?
    return image, slopes

def Pos_estimation(slopes):
    if slopes[10] > -0.3 and slopes[10] < 0.3:
        print("서있음")
    elif slopes[10] < -0.3 or slopes[10] > 0.3:
        print("기울어짐")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    input_tensor = preprocess(frame)

    with torch.no_grad():
        output = model(input_tensor)

    if len(output) > 0:
        output = output[0]
        scores = output['scores'].cpu().numpy()
        high_scores_idx = np.where(scores > 0.9)[0]

        if len(high_scores_idx) > 0:
            keypoints = output['keypoints'][high_scores_idx].cpu().numpy()
            frame_with_keypoints, slopes = draw_keypoints(frame, keypoints)
            cv2.imshow('Skeleton Detection', frame_with_keypoints)
            Pos_estimation(slopes)
        else:
            cv2.imshow('Skeleton Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
