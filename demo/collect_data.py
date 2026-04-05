import cv2
import mediapipe as mp
import csv

# --- CẤU HÌNH ---
LABEL = 9  # Thay đổi từ 0-9 khi thu thập các số tương ứng
FILE_NAME = 'hand_data.csv'

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
print(f"Đang thu thập số: {LABEL}. Nhấn giữ 's' để lưu, 'q' để thoát.")

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    
    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Trích xuất 21 điểm (x, y) = 42 giá trị
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y])
            
            # Nhấn 's' để ghi vào CSV
            if cv2.waitKey(1) & 0xFF == ord('s'):
                with open(FILE_NAME, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([LABEL] + landmarks)
                print(f"Đã lưu mẫu cho số {LABEL}")

    cv2.imshow("Collect Data - Dong A University", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()