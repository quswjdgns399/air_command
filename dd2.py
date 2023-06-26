from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout
from PyQt5.QtGui import QFont
import cv2
import mediapipe as mp
import subprocess
import numpy as np
import time, os
import sys
from PyQt5.uic import loadUi
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model
from keras.utils import to_categorical

class AppUI(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('남자 Eye들')
        self.setGeometry(300, 300, 300, 150)
        self.setFont(QFont('Arial', 12))
        self.create_time = None

        # Motion Detection Activation Button
        self.motion_detection_button = QPushButton('모션 인식 암호 등록', self)
        self.motion_detection_button.move(20, 50)
        self.motion_detection_button.setFixedSize(300, 30)
        self.motion_detection_button.clicked.connect(self.activate_motion_detection)
        self.test_ui = None

        self.unlock_button = QPushButton('모션 암호 확인', self)
        self.unlock_button.move(10, 100)
        self.unlock_button.setFixedSize(300, 30)
        self.unlock_button.clicked.connect(self.unlock_password)

        # Vertical Layout Configuration
        layout = QVBoxLayout()
        layout.addWidget(self.motion_detection_button)

        self.setLayout(layout)

    # Function to activate motion detection
    def activate_motion_detection(self):
        import cv2
        import mediapipe as mp
        import numpy as np
        import time, os

        actions = ['ac1', 'ac2', 'ac3']
        seq_length = 30
        secs_for_action = 30

        # MediaPipe hands model
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        hands = mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)

        cap = cv2.VideoCapture(0)

        created_time = int(time.time())
        os.makedirs('dataset', exist_ok=True)

        while cap.isOpened():
            for idx, action in enumerate(actions):
                data = []

                ret, img = cap.read()

                img = cv2.flip(img, 1)

                cv2.putText(img, f'Waiting for collecting {action.upper()} action...', org=(10, 30),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
                cv2.imshow('img', img)
                cv2.waitKey(2000)

                start_time = time.time()

                while time.time() - start_time < secs_for_action:
                    ret, img = cap.read()

                    img = cv2.flip(img, 1)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    result = hands.process(img)
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                    if result.multi_hand_landmarks is not None:
                        for res in result.multi_hand_landmarks:
                            joint = np.zeros((21, 4))
                            for j, lm in enumerate(res.landmark):
                                joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

                            # Compute angles between joints
                            v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19],
                                 :3]  # Parent joint
                            v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                                 :3]  # Child joint
                            v = v2 - v1  # [20, 3]
                            # Normalize v
                            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                            # Get angle using arcos of dot product
                            angle = np.arccos(np.einsum('nt,nt->n',
                                                        v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                                        v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19],
                                                        :]))  # [15,]

                            angle = np.degrees(angle)  # Convert radian to degree

                            angle_label = np.array([angle], dtype=np.float32)
                            angle_label = np.append(angle_label, idx)

                            d = np.concatenate([joint.flatten(), angle_label])

                            data.append(d)

                            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

                    cv2.imshow('img', img)
                    if cv2.waitKey(1) == ord('q'):
                        break

                data = np.array(data)

                # Create sequence data
                full_seq_data = []
                for seq in range(len(data) - seq_length):
                    full_seq_data.append(data[seq:seq + seq_length])

                full_seq_data = np.array(full_seq_data)
                print(action, full_seq_data.shape)
                np.save(os.path.join('dataset', f'seq_{action}'), full_seq_data)
            cap.release()
            cv2.destroyAllWindows()
        self.train_model()

    def train_model(self):
        os.environ['CUDA_VISIBLE_DEVICES'] = '1'
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

        actions = [
            'ac1',
            'ac2',
            'ac3'
        ]

        try:
            data = np.concatenate([np.load(os.path.join('dataset', f'seq_ac1.npy')),
                                   np.load(os.path.join('dataset', f'seq_ac2.npy')),
                                   np.load(os.path.join('dataset', f'seq_ac3.npy'))], axis=0)
            x_data = data[:, :, :-1]
            labels = data[:, 0, -1]

            print(x_data.shape)
            print(labels.shape)

            y_data = to_categorical(labels, num_classes=len(actions))

            x_data = x_data.astype(np.float32)
            y_data = y_data.astype(np.float32)

            x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.1, random_state=2021)

            print(x_train.shape, y_train.shape)
            print(x_val.shape, y_val.shape)

            model = Sequential([
                LSTM(64, activation='relu', input_shape=x_train.shape[1:]),
                Dense(32, activation='relu'),
                Dense(len(actions), activation='softmax')
            ])

            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
            model.summary()

            checkpoint = ModelCheckpoint('new_model.h5', monitor='val_acc', verbose=1, save_best_only=True,
                                         mode='auto')
            reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=50, verbose=1, mode='auto')

            history = model.fit(
                x_train,
                y_train,
                validation_data=(x_val, y_val),
                epochs=200,
                callbacks=[checkpoint, reduce_lr]
            )

            fig, loss_ax = plt.subplots(figsize=(16, 10))
            acc_ax = loss_ax.twinx()

            loss_ax.plot(history.history['loss'], 'y', label='train loss')
            loss_ax.plot(history.history['val_loss'], 'r', label='val loss')
            loss_ax.set_xlabel('epoch')
            loss_ax.set_ylabel('loss')
            loss_ax.legend(loc='upper left')

            acc_ax.plot(history.history['acc'], 'b', label='train acc')
            acc_ax.plot(history.history['val_acc'], 'g', label='val acc')
            acc_ax.set_ylabel('accuracy')
            acc_ax.legend(loc='upper left')

            plt.show()

            model = load_model('new_model.h5')

            y_pred = model.predict(x_val)

            y_val_labels = np.argmax(y_val, axis=1)
            y_pred_labels = np.argmax(y_pred, axis=1)

            confusion_matrix(y_val_labels, y_pred_labels)

            if len(y_pred) == 1:
                conf = y_pred[0]
            else:
                i_pred = int(np.argmax(y_pred))
                conf = y_pred[i_pred]
        except FileNotFoundError:
            print(f"Error: 파일을 찾을수 없습니다.")
        except Exception as e:
            print(f"Error: {str(e)}")
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
            subprocess.Popen(["python", "11.py"])  # 여기에 실제 Python 파일 이름을 입력하세요
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

                if this_action == 'ac1':
                    if action_counter.get(this_action, 0) >= count_threshold and ac1_count < 10:
                        ac1_count += 10
                        if ac1_count == 10:
                            cv2.putText(img, f'Waiting for next action...', org=(10, 30),
                                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255),
                                        thickness=2)
                            cv2.imshow('img', img)
                        action_counter = {}

                elif this_action == 'ac2':
                    if action_counter.get(this_action, 0) >= count_threshold and ac2_count < 10:
                        ac2_count += 10
                        if ac2_count == 10:
                            cv2.putText(img, f'Waiting for next action...', org=(10, 30),
                                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255),
                                        thickness=2)
                            cv2.imshow('img', img)
                        action_counter = {}

                elif this_action == 'ac3':
                    if action_counter.get(this_action, 0) >= count_threshold and ac3_count < 10:
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