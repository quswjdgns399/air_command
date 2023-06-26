import sys
import subprocess
from PyQt5 import uic, QtCore
from PyQt5.QtWidgets import QApplication, QMainWindow

form_class = uic.loadUiType("./testBtn.ui")[0]


class WindowClass(QMainWindow, form_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.testBtn.clicked.connect(self.toggle_exe)
        self.testBtn2.clicked.connect(self.toggle2_exe)
        self.testBtn3.clicked.connect(self.toggle3_exe)
        self.proc = None

    def toggle_exe(self):
        if self.proc is None:
            self.start_exe()
        else:
            self.stop_exe()

    def start_exe(self):
        if self.proc is None:
            self.statusBar().showMessage("음성인식 파일 실행중...")
            self.proc = subprocess.Popen(["python", "C:\\main_ui\\voicecommand_final.py"])
            self.testBtn.setText("음성인식 종료")  # 버튼 텍스트 변경

    def stop_exe(self):
        if self.proc is not None:
            subprocess.run(['taskkill', '/f', '/t', '/pid', str(self.proc.pid)])
            self.proc = None
            self.testBtn.setText("음성인식 실행")  # 버튼 텍스트 변경
            self.statusBar().showMessage("음성인식 종료")

    def toggle2_exe(self):
        if self.proc is None:
            self.start2_exe()
        else:
            self.stop2_exe()

    def start2_exe(self):
        if self.proc is None:
            self.statusBar().showMessage("모션인식 파일 실행중...")
            self.proc = subprocess.Popen(["python", "C:\main_ui\motion_final.py"])
            self.testBtn2.setText("모션인식 종료")  # 버튼 텍스트 변경

    def stop2_exe(self):
        if self.proc is not None:
            subprocess.run(['taskkill', '/f', '/t', '/pid', str(self.proc.pid)])
            self.proc = None
            self.testBtn2.setText("모션인식 실행")  # 버튼 텍스트 변경
            self.statusBar().showMessage("모션인식 종료")

    def toggle3_exe(self):
        if self.proc is None:
            self.start3_exe()
        else:
            self.stop3_exe()

    def start3_exe(self):
        if self.proc is None:
            self.statusBar().showMessage("아이트레킹 파일 실행중...")
            self.proc = subprocess.Popen(["python", "C:\main_ui\eyetrac_final.py"])
            self.testBtn3.setText("아이트레킹 종료")  # 버튼 텍스트 변경

    def stop3_exe(self):
        if self.proc is not None:
            subprocess.run(['taskkill', '/f', '/t', '/pid', str(self.proc.pid)])
            self.proc = None
            self.testBtn3.setText("아이트레킹 실행")  # 버튼 텍스트 변경
            self.statusBar().showMessage("아이트레킹 종료")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWindow = WindowClass()
    myWindow.show()
    sys.exit(app.exec_())