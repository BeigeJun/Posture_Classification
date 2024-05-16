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

csv_file = 'captured_images/pos_data.csv'

def read_lines(path):
    keypoints = []
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
    return keypoints


def make_center_pos(key, num):
    np_key = np.array(key)
    center_x = np.mean(np_key[num, :, 0])
    center_y = np.mean(np_key[num, :, 1])
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


points = read_lines(csv_file)

center = make_center_pos(points, 0)
changed_pos = remake_pos(points, center)
original_keypoints = np.array(points)
changed_keypoints = np.array(changed_pos)

print(changed_pos)

plt.figure(figsize=(10, 5))

for i in range(len(original_keypoints)):
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
