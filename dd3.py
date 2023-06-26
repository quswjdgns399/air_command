from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel
from PyQt5.QtGui import QFont
import cv2
import mediapipe as mp
import subprocess
import numpy as np
import sys
from tensorflow.keras.models import load_model

class AppUI(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('남자 Eye들')
        self.setGeometry(300, 300, 300, 150)
        self.setFont(QFont('Arial', 12))
        self.create_time = None
        self.label = QLabel("남자EYE들", self)
        self.label.setObjectName("label")
        self.label.setGeometry(70, 40, 250, 30)
        self.label.setFont(QFont("Arial", 20))
        self.create_time = None

        self.unlock_button = QPushButton('모션 암호 확인', self)
        self.unlock_button.move(10, 100)
        self.unlock_button.setFixedSize(300, 30)
        self.unlock_button.clicked.connect(self.unlock_password)
    def unlock_password(self):
        actions = ['ac1','ac2','ac3']
        seq_length = 30

        model = load_model('new_model.h5')

        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        hands = mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)

        cap = cv2.VideoCapture(0)

        seq = []
        action_seq = []
        action_counter = {}
        ac1_count = 0
        ac2_count = 0
        ac3_count = 0
        sum_of_count = 0
        count_threshold = 50  # 변경된 부분: 지속시간 대신 카운트 임계값 설정

        def open_file():
            subprocess.Popen(["python", "main_ui.py"])  # 여기에 실제 Python 파일 이름을 입력하세요
            sys.exit()

        while cap.isOpened():
            ret, img = cap.read()
            img0 = img.copy()

            img = cv2.flip(img, 1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = hands.process(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            if result.multi_hand_landmarks is not None:
                for res in result.multi_hand_landmarks:
                    joint = np.zeros((21, 4))
                    for j, lm in enumerate(res.landmark):
                        joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

                    v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :3]
                    v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :3]
                    v = v2 - v1
                    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                    angle = np.arccos(np.einsum('nt,nt->n',
                                                v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                                v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))

                    angle = np.degrees(angle)

                    d = np.concatenate([joint.flatten(), angle])

                    seq.append(d)

                    mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

                    if len(seq) < seq_length:
                        continue

                    input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)
                    y_pred = model.predict(input_data).squeeze()

                    i_pred = int(np.argmax(y_pred))
                    conf = y_pred[i_pred]

                    if conf < 0.9:
                        continue

                    action = actions[i_pred]
                    action_seq.append(action)

                    if len(action_seq) < 3:
                        continue

                    this_action = '?'
                    if action_seq[-1] == action_seq[-2] == action_seq[-3]:
                        this_action = action
                        action_counter[this_action] = action_counter.get(this_action, 0) + 1
                    else:
                        action_counter = {}

                    if this_action == 'ac1' and action_counter.get(this_action, 0) >= count_threshold and ac1_count < 10:
                        ac1_count += 10
                        if ac1_count == 10:
                            cv2.putText(img, f'Waiting for next action...', org=(10, 30),
                                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255),
                                        thickness=2)
                            cv2.imshow('img', img)
                        action_counter = {}
                    if this_action == 'ac2' and action_counter.get(this_action, 0) >= count_threshold and ac2_count < 10:
                        ac2_count += 10
                        if ac2_count == 10:
                            cv2.putText(img, f'Waiting for next action...', org=(10, 30),
                                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255),
                                        thickness=2)
                            cv2.imshow('img', img)
                        action_counter = {}
                    if this_action == 'ac3' and action_counter.get(this_action, 0) >= count_threshold and ac3_count < 10:
                        ac3_count += 10
                        if ac3_count == 10:
                            cv2.putText(img, f'Waiting for next action...', org=(10, 30),
                                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255),
                                        thickness=2)
                            cv2.imshow('img', img)
                        action_counter = {}

                    sum_of_count = ac1_count + ac2_count + ac3_count

                    if sum_of_count >= 30:
                        open_file()

                    cv2.putText(img, f'{this_action.upper()}',
                                org=(int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0] + 20)),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

            cv2.imshow('img', img)
            if cv2.waitKey(1) == ord('q'):
                break

if __name__ == '__main__':
    app = QApplication([])
    app_ui = AppUI()
    app_ui.show()
    app.exec_()