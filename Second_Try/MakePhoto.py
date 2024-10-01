import cv2
import os

# 비디오 파일 경로
video_path = 'C:/Users/wns20/PycharmProjects/SMART_CCTV/5th_Try/Data/video.mp4'
cap = cv2.VideoCapture(video_path)

# 이미지 저장 폴더 경로
output_dir = 'C:/Users/wns20/PycharmProjects/SMART_CCTV/Second_Try/Frames'

# 클래스별 폴더 미리 생성
classes = ['Stand', 'Sit_chair', 'Sit_floor', 'FallingDown', 'FallDown', 'Terrified']
for cls in classes:
    os.makedirs(os.path.join(output_dir, cls), exist_ok=True)

# 클래스별 프레임 카운터 저장
frame_counts = {cls: 0 for cls in classes}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow('Video', frame)

    key = cv2.waitKey(1)
    if key in [ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6')]:
        # 키에 따른 클래스 설정
        if key == ord('1'):
            label = 'Stand'
        elif key == ord('2'):
            label = 'Sit_chair'
        elif key == ord('3'):
            label = 'Sit_floor'
        elif key == ord('4'):
            label = 'FallingDown'
        elif key == ord('5'):
            label = 'FallDown'
        elif key == ord('6'):
            label = 'Terrified'

        # 해당 클래스에 대한 프레임 카운트를 증가
        frame_counts[label] += 1

        # 이미지 파일 경로 설정
        output_path = os.path.join(output_dir, label, f'frame_{frame_counts[label]}.jpg')

        # 이미지 저장
        cv2.imwrite(output_path, frame)
        print(f"Saved frame to {output_path} with label: {label}")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
