import cv2
import os
import time

# 비디오 파일 경로
video_path = '/5th_Try/Data/Falling.mp4'
cap = cv2.VideoCapture(video_path)

# 이미지 저장 폴더 경로
output_dir = '/Second_Try/Frames'

# 클래스별 폴더 미리 생성
classes = ['Stand', 'Sit_chair', 'Sit_floor', 'FallingDown', 'FallDown', 'Sleep']
for cls in classes:
    os.makedirs(os.path.join(output_dir, cls), exist_ok=True)

# 클래스별 프레임 카운터 저장
frame_counts = {cls: 0 for cls in classes}

# 캡쳐 시작 여부를 저장하는 변수
is_capturing = False
label = None
last_capture_time = time.time()  # 마지막으로 프레임을 저장한 시간

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow('Video', frame)

    # 현재 시간과 마지막 캡쳐 시간 차이를 계산
    current_time = time.time()

    # 0.1초마다 프레임을 저장
    if is_capturing and (current_time - last_capture_time >= 0.1):
        # 이미지 파일 경로 설정
        frame_counts[label] += 1
        output_path = os.path.join(output_dir, label, f'frame_{frame_counts[label]}.jpg')

        # 이미지 저장
        cv2.imwrite(output_path, frame)
        print(f"Saved frame to {output_path} with label: {label}")

        # 마지막 캡쳐 시간을 업데이트
        last_capture_time = current_time

    key = cv2.waitKey(1)

    # 라벨 선택 및 캡쳐 시작
    if key in [ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6')]:
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
            label = 'Sleep'

        is_capturing = True  # 캡쳐 시작
        last_capture_time = time.time()  # 캡쳐 시간 초기화
        print(f"Started capturing with label: {label}")

    # 캡쳐 중지
    elif key == ord('s'):
        is_capturing = False  # 캡쳐 중지
        print("Capturing stopped")

    # 프로그램 종료
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
