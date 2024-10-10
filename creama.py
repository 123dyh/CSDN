import argparse
import random
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QComboBox, QLabel, QWidget, QHBoxLayout
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
from torch.backends import cudnn

import dswrapper as ds
from threading import Thread
from PIL import Image
import time
from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np
import cv2
import torch
from models.experimental import attempt_load
from utils.augmentations import letterbox
from utils.general import non_max_suppression, scale_coords, check_img_size
from utils.plots import plot_one_box
from utils.torch_utils import select_device
import os


class Camera():
    def __init__(self, base_save_path, open_index, cam_type, camera_name):
        self.runflag = True
        self.base_save_path = base_save_path
        self.open_index = open_index
        self.cam_type = cam_type
        self.camera_name = camera_name
        self.raw_image = None
        self.setup_camera()

    def setup_camera(self):
        num = ds.ViOpenDalsaDeviceGX(self.open_index)
        # 添加其他相机初始化代码，具体取决于相机型号和 SDK

    def run(self):
        while self.runflag:
            num = ds.ViStartAcqSingleFrame(self.open_index, self.cam_type)
            time.sleep(0.001)
            self.raw_image = ds.get_imaging(self.open_index, self.cam_type)
            if self.raw_image is not None:
                numpy_image = self.raw_image.get_numpy_array()
                if numpy_image is not None:
                    img = Image.fromarray(numpy_image, 'L')
                    # 创建主文件夹
                    main_save_path = os.path.join(self.base_save_path, "Dalsa_Cameras")
                    os.makedirs(main_save_path, exist_ok=True)
                    # 创建子文件夹
                    save_path = os.path.join(main_save_path, self.camera_name)
                    os.makedirs(save_path, exist_ok=True)
                    # 保存图片
                    img.save(os.path.join(save_path, f"{time.time()}.jpg"))

    def stop(self):
        self.runflag = False
        print("停止")

    def set_start(self):
        self.runflag = True

class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        # self.setupUi()
        self.device = '0'
        self.cameras = []
        self.init_cameras()
        self.init_ui()

    def init_cameras(self):
        camera_names = ["cam1", "cam2", "cam3"]
        base_save_path = "D:/333"  # 更改为您的基本文件夹路径
        for i, camera_name in enumerate(camera_names):
            camera = Camera(base_save_path, open_index=i, cam_type=0, camera_name=camera_name)
            camera.set_start()
            camera_thread = Thread(target=camera.run)
            camera_thread.start()
            self.cameras.append(camera)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.show_video_frame)
        self.timer.start(30)

    def init_ui(self):
        self.setWindowTitle("Camera Display")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(9, -1, 781, 461))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtWidgets.QLabel(self.horizontalLayoutWidget)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(10, 460, 781, 80))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.pushButton = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.pushButton.setObjectName("pushButton")
        self.verticalLayout.addWidget(self.pushButton)
        self.pushButton_2 = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.pushButton_2.setObjectName("pushButton_2")
        self.verticalLayout.addWidget(self.pushButton_2)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "显示"))
        self.pushButton.setText(_translate("MainWindow", "开始"))
        self.pushButton.clicked.connect(self.start_cameras)
        self.pushButton_2.setText(_translate("MainWindow", "停止"))
        self.pushButton_2.clicked.connect(self.stop_cameras)

    def start_cameras(self):
        for camera in self.cameras:
            camera.set_start()

    def stop_cameras(self):
        for camera in self.cameras:
            camera.stop()

    def show_video_frame(self):
        name_list = []
        for i, camera in enumerate(self.cameras):
            if camera.raw_image is not None:
                numpy_image = camera.raw_image.get_numpy_array()
                if numpy_image is not None:
                    cur_frame = np.array(numpy_image, dtype=np.uint8)
                    img = cv2.cvtColor(cur_frame, cv2.COLOR_GRAY2RGB)
                    showimg = img
                    show = cv2.resize(showimg, (640, 480))
                    result = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
                    show_image = QImage(result.data, result.shape[1], result.shape[0], QImage.Format_RGB888)
                    self.label.setPixmap(QPixmap.fromImage(show_image))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())
