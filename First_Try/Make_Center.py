import csv
import numpy as np

csv_file = 'C:/Users/wns20/PycharmProjects/SMART_CCTV/captured_images/pos_data.csv'
remake_csv_file = 'C:/Users/wns20/PycharmProjects/SMART_CCTV/captured_images/pos_data_remake.csv'

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
for i in range(len(points)):
    center = make_center_pos(points, i)
    changed_pos = remake_pos(points, center)
    original_keypoints = np.array(points)
    changed_keypoints = np.array(changed_pos)
    print(changed_keypoints)
    # plt.figure(figsize=(10, 5))
    # plt.subplot(1, 2, 1)
    # plt.scatter(original_keypoints[i, :, 0], original_keypoints[i, :, 1], color='blue')
    # plt.title('Original Keypoints')
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.grid(True)
    #
    # plt.subplot(1, 2, 2)
    # plt.scatter(changed_keypoints[i, :, 0], changed_keypoints[i, :, 1], color='red')
    # plt.title('Changed Keypoints')
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.grid(True)
    #
    # plt.tight_layout()
    # plt.show()
# print(lines)
# print(original_keypoints)
# print(original_keypoints.shape)



with open(remake_csv_file, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    print(changed_keypoints.shape)
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