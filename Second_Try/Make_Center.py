import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
import os
csv_file = 'C:/Users/wns20/PycharmProjects/SMART_CCTV/captured_images/pos_data.csv'
remake_csv_file = 'C:/Users/wns20/PycharmProjects/SMART_CCTV/captured_images/pos_data_remake.csv'

codes = [
    Path.MOVETO,
    Path.LINETO,
    Path.LINETO
]

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
    Max = 0
    Min = 10000000000000
    new_keypoints = []
    for row in keypoints:
        row_keypoints = []
        for pos in row:
            new_pos = [pos[0] - center_pos[0] + 1000, pos[1] - center_pos[1] + 1000]
            row_keypoints.append(new_pos)
            if new_pos[0] < Min:
                Min = new_pos[0]
            if new_pos[1] < Min:
                Min = new_pos[1]
            if new_pos[0] > Max:
                Max = new_pos[0]
            if new_pos[1] > Max:
                Max = new_pos[1]
        new_keypoints.append(row_keypoints)
    for row in new_keypoints:
        for pos in row:
            pos[0] = (pos[0] - Min) / (Max - Min)
            pos[1] = (pos[1] - Min) / (Max - Min)
    with open('Max_Min.txt', 'w') as file:
        file.write(f"{Max}\n")
        file.write(f"{Min}\n")
    return new_keypoints

points, labels, lines = read_lines(csv_file)

fieldnames = []
fieldnames.append(f'keypoint_5_x')
fieldnames.append(f'keypoint_5_y')
fieldnames.append(f'keypoint_6_x')
fieldnames.append(f'keypoint_6_y')
for i in range(11, 17):
    fieldnames.append(f'keypoint_{i}_x')
    fieldnames.append(f'keypoint_{i}_y')
fieldnames.append('label')
print(fieldnames)

output_folder = 'C:/Users/wns20/PycharmProjects/SMART_CCTV/captured_images/output_images'
labeled_0 = 'C:/Users/wns20/PycharmProjects/SMART_CCTV/Second_Try/Train/0'
labeled_1 = 'C:/Users/wns20/PycharmProjects/SMART_CCTV/Second_Try/Train/1'
cnt_label_0 = 0
cnt_label_1 = 0
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for i in range(len(points)):
    center = make_center_pos(points, i)
    changed_pos = remake_pos(points, center)
    original_keypoints = np.array(points)
    changed_keypoints = np.array(changed_pos)
    print(changed_keypoints)
    plt.figure(figsize=(3, 5))

    plt.scatter(changed_keypoints[i, 5:17, 0], changed_keypoints[i, 5:17, 1], color='red')
    head_x = (changed_keypoints[i, 3, 0] + changed_keypoints[i, 4, 0]) / 2
    head_y = (changed_keypoints[i, 3, 1] + changed_keypoints[i, 4, 1]) / 2
    plt.scatter(head_x, head_y, color='red', s=500)

    ax = plt.gca()
    ax.axis('off')
    ax.invert_xaxis()
    ax.invert_yaxis()

    shoulder_verts = changed_keypoints[i, [5, 6], :2]
    shoulder_path = Path(shoulder_verts, [Path.MOVETO, Path.LINETO])
    shoulder_line = patches.PathPatch(shoulder_path, linewidth=2, facecolor='none', edgecolor='red')
    ax.add_patch(shoulder_line)

    for j in range(2):
        body_verts = changed_keypoints[i, [5 + j, 11 + j], :2]
        body_path = Path(body_verts, [Path.MOVETO, Path.LINETO])
        body_line = patches.PathPatch(body_path, linewidth=2, facecolor='none', edgecolor='red')
        ax.add_patch(body_line)

    pelvis_verts = changed_keypoints[i, [11, 12], :2]
    pelvis_path = Path(pelvis_verts, [Path.MOVETO, Path.LINETO])
    pelvis_line = patches.PathPatch(pelvis_path, linewidth=2, facecolor='none', edgecolor='red')
    ax.add_patch(pelvis_line)

    for j in range(2):
        verts = changed_keypoints[i, [5 + j, 7 + j, 9 + j], :2]
        path = Path(verts, codes)
        line = patches.PathPatch(path, linewidth=2, facecolor='none', edgecolor='red')
        ax.add_patch(line)

    for j in range(2):
        verts = changed_keypoints[i, [11 + j, 13 + j, 15 + j], :2]
        path = Path(verts, codes)
        line = patches.PathPatch(path, linewidth=2, facecolor='none', edgecolor='red')
        ax.add_patch(line)

    plt.tight_layout()
    if labels[i] == 'Sit':
        plt.savefig(os.path.join(labeled_0, f'image_{cnt_label_0}.png'))
        cnt_label_0 += 1
    elif labels[i] == 'Stand':
        plt.savefig(os.path.join(labeled_1, f'image_{cnt_label_1}.png'))
        cnt_label_1 += 1
    plt.close()


with open(remake_csv_file, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()

    for cnt in range(lines):
        row = {}
        row[f'keypoint_5_x'] = changed_keypoints[cnt][5][0]
        row[f'keypoint_5_y'] = changed_keypoints[cnt][5][1]
        row[f'keypoint_6_x'] = changed_keypoints[cnt][6][0]
        row[f'keypoint_6_y'] = changed_keypoints[cnt][6][1]
        for i in range(11, 17):
            row[f'keypoint_{i}_x'] = changed_keypoints[cnt][i][0]
            row[f'keypoint_{i}_y'] = changed_keypoints[cnt][i][1]
        row['label'] = labels[cnt]
        writer.writerow(row)
