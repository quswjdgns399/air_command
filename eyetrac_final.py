import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import keyboard

pyautogui.FAILSAFE = False

pyautogui.PAUSE = 0.055 # 마우스 이동 속도 제어 (흔들림 제어)
cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True) # 얼굴 그물망
screen_w, screen_h = pyautogui.size()
prev_x, prev_y = 0, 0
step = 30 # amount to move mouse by

# 화면 중앙 좌표 계산q
initial_mouse_pos = pyautogui.position()

# 초기 마우스 위치를 화면 중앙으로 이동

pyautogui.moveTo(screen_w // 2, screen_h // 2)
while True:
    _, frame = cam.read()
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb_frame)
    landmarks_points = output.multi_face_landmarks
    frame_h, frame_w, _ = frame.shape
    if landmarks_points:
        landmarks = landmarks_points[0].landmark

        # Find left eye landmarks
        left_eye_landmarks = landmarks[469:473]
        left_eye_center = np.mean(np.array([(l.x * frame_w, l.y * frame_h) for l in left_eye_landmarks]), axis=0)

        # Find right eye landmarks
        right_eye_landmarks = landmarks[474:478]
        right_eye_center = np.mean(np.array([(l.x * frame_w, l.y * frame_h) for l in right_eye_landmarks]), axis=0)

        # Move mouse based on eye positions
        move_x = int((left_eye_center[0] + right_eye_center[0]) / 2 - prev_x)
        move_y = int((left_eye_center[1] + right_eye_center[1]) / 2 - prev_y)
        pyautogui.move(move_x * 10, move_y * 10, duration=0.1)
        prev_x, prev_y = (left_eye_center[0] + right_eye_center[0]) / 2, (left_eye_center[1] + right_eye_center[1]) / 2

        left = [landmarks[145], landmarks[159]]
        for landmark in left:
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 255))
        if (left[0].y - left[1].y) < 0.005:
            pyautogui.click()
            pyautogui.sleep(1)

    key = cv2.waitKey(1)
    if keyboard.is_pressed('q'):
        break
    elif keyboard.is_pressed('up'):  # Up arrow key
        pyautogui.move(0, -step)
    elif keyboard.is_pressed('down'):  # Down arrow key
        pyautogui.move(0, step)
    elif keyboard.is_pressed('left'):  # Left arrow key
        pyautogui.move(-step, 0)
    elif keyboard.is_pressed('right'):  # Right arrow key
        pyautogui.move(step, 0)

    #cv2.imshow('Eye Controlled Mouse', frame)

cv2.destroyAllWindows()
cam.release()