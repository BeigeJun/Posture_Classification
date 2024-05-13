import cv2
import os

video_path = 'output.avi'

if not os.path.isfile(video_path):
    print(f'Error: {video_path} not found')
    exit()

cap = cv2.VideoCapture(video_path)

save_directory = 'captured_images'
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
