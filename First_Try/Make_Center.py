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
import numpy as np

csv_file = 'C:/Users/wns20/PycharmProjects/SMART_CCTV/captured_images/pos_data.csv'

def read_lines(path):
    keypoints = []
    labels = []
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
    return keypoints, labels


def make_center_pos(key, num):
    np_key = np.array(key)
    max_x = np.max(np_key[num, :, 0])
    min_x = np.min(np_key[num, :, 0])
    max_y = np.max(np_key[num, :, 1])
    min_y = np.min(np_key[num, :, 1])
    center_x = (max_x + min_x) / 2
    center_y = (max_y + min_y) / 2
    center_pos = [center_x, center_y]
    return center_pos

def remake_pos(keypoints, center_pos):
    new_keypoints = []
    for row in keypoints:
        row_keypoints = []
        for pos in row:
            new_pos = [pos[0] - center_pos[0], pos[1] - center_pos[1]]
            row_keypoints.append(new_pos)
        new_keypoints.append(row_keypoints)
    return new_keypoints


points, labels = read_lines(csv_file)

fieldnames = []
for i in range(17):
    fieldnames.append(f'keypoint_{i + 1}_x')
    fieldnames.append(f'keypoint_{i + 1}_y')
fieldnames.append('label')

for i in range(len(points)):
    center = make_center_pos(points, i)
    changed_pos = remake_pos(points, center)
    original_keypoints = np.array(points)
    changed_keypoints = np.array(changed_pos)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(original_keypoints[i, :, 0], original_keypoints[i, :, 1], color='blue')
    plt.title('Original Keypoints')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.scatter(changed_keypoints[i, :, 0], changed_keypoints[i, :, 1], color='red')
    plt.title('Changed Keypoints')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# with open(csv_file, mode='w', newline='') as file:
#     writer = csv.DictWriter(file, fieldnames=fieldnames)
#     writer.writeheader()
#
#     for cnt in range(len(file_list) - 2):
#         row = {}
#         for i in range(len(Sceleton_1)):
#             row[f'keypoint_{i + 1}_x'] = Sceleton_1[i][0]
#             row[f'keypoint_{i + 1}_y'] = Sceleton_1[i][1]
#         row['label'] = label
#         writer.writerow(row)
