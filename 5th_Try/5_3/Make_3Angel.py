import csv
import numpy as np
import os
csv_file = 'C:/Users/wns20/PycharmProjects/SMART_CCTV/5th_Try/Data/pos_data.csv'
csv_Angle_path = 'C:/Users/wns20/PycharmProjects/SMART_CCTV/5th_Try/Data/3Angle_data.csv'

def read_lines(path):
    keypoints = []
    labels = []
    count = 0
    with open(path, mode='r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            keypoints_row = []
            for i in range(1, 18):
                x = float(row[f'keypoint_{i}_x'])
                y = float(row[f'keypoint_{i}_y'])
                keypoints_row.append([x, y])
            label = row['label']
            keypoints.append(keypoints_row)
            labels.append(label)
            count += 1
    return keypoints, labels, count


def make_angle(point1, point2):
    dx = point1[0] - point2[0]
    dy = point1[1] - point2[1]
    if dx != 0:
        slope = dy / dx
    else:
        slope = 0
    return slope


points, labels, lines = read_lines(csv_file)
print(points)
cnt_label_0 = 0
cnt_label_1 = 0
cnt_label_2 = 0
keypoint_names = {
    0: 'nose',
    1: 'left_eye',
    2: 'right_eye',
    3: 'left_ear',
    4: 'right_ear',
    5: 'left_shoulder',
    6: 'right_shoulder',
    7: 'left_elbow',
    8: 'right_elbow',
    9: 'left_wrist',
    10: 'right_wrist',
    11: 'left_hip',
    12: 'right_hip',
    13: 'left_knee',
    14: 'right_knee',
    15: 'left_ankle',
    16: 'right_ankle',
    17: 'neck',
    18: 'left_palm',
    19: 'right_palm',
    20: 'spine2(back)',
    21: 'spine1(waist)',
    22: 'left_instep',
    23: 'right_instep'
}


fieldnames = []
fieldnames.append(keypoint_names[5] + "to" + keypoint_names[12])#왼쪽 어깨 -> 왼쪽 골반
fieldnames.append(keypoint_names[6] + "to" + keypoint_names[11])#오른쪽 어깨 -> 오른쪽 골반
fieldnames.append(keypoint_names[5]+keypoint_names[6] + "to" + keypoint_names[11] + keypoint_names[12])
fieldnames.append('label')

with open(csv_Angle_path, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()

    for cnt in range(lines):
        row = {}
        row[keypoint_names[5] + "to" + keypoint_names[12]] = make_angle(points[cnt][5], points[cnt][11])
        row[keypoint_names[6] + "to" + keypoint_names[11]] = make_angle(points[cnt][5], points[cnt][7])
        Middle_shoulder = [(points[cnt][5][0] + points[cnt][6][0])/2, (points[cnt][5][1] + points[cnt][6][1])/2]
        Middle_pelvis = [(points[cnt][11][0] + points[cnt][12][0])/2, (points[cnt][11][1] + points[cnt][12][1])/2]
        row[keypoint_names[5]+keypoint_names[6] + "to" + keypoint_names[11] + keypoint_names[12]] = make_angle(Middle_shoulder, Middle_pelvis)
        if labels[cnt] == 'Stand' or labels[cnt] == 'Sit_chair' or labels[cnt] == 'Sit_floor':
            row['label'] = 'straight'
        else:
            row['label'] = 'tilt'
        writer.writerow(row)
