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
from untitled2 import Ui_MainWindow
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
            time.sleep(0)
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

class MainWindow(QMainWindow,Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.device = '0'
        self.cameras = []
        self.init_models()
        self.init_cameras()
        self.showFullScreen()

    def init_models(self):
        # 添加你的模型初始化代码，保留原有的model_init函数或将其整合到这里
        # ...
        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', nargs='+', type=str,
                            default='E:/ruanjian/best.pt',
                            help='model path(s)')
        parser.add_argument('--source', type=str, default='data/images', help='file/dir/URL/glob, 0 for webcam')
        parser.add_argument('--data', type=str, default='data/coco128.yaml', help='(optional) dataset.yaml path')
        parser.add_argument('--img-size', nargs='+', type=int, default=640,
                            help='inference size h,w')
        parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
        parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
        parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
        parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--view-img', action='store_true', help='show results')
        parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
        parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
        parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
        parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
        parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
        parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
        parser.add_argument('--augment', action='store_true', help='augmented inference')
        parser.add_argument('--visualize', action='store_true', help='visualize features')
        parser.add_argument('--update', action='store_true', help='update all models')
        parser.add_argument('--project', default='runs/detect', help='save results to project/name')
        parser.add_argument('--name', default='exp', help='save results to project/name')
        parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
        parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
        parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
        parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
        # 解析命令行参数，并将结果存储在 self.opt 中。打印解析后的参数。
        self.opt = parser.parse_args()
        print(self.opt)
        # 默认使用’--weights‘中的权重来进行初始化
        source, weights, view_img, save_txt, imgsz = self.opt.source, self.opt.weights, self.opt.view_img, self.opt.save_txt, self.opt.img_size
        # 如果openfile_name_model不为空，则使用openfile_name_model权重进行初始化

        # weights = self.model

        self.device = select_device(self.opt.device)
        self.half = self.device.type != 'cpu'
        # 提高模型的运行效率。
        cudnn.benchmark = True

        # lode model
        # 这将载入模型的权重，这些权重将用于后续的操作。
        self.model = attempt_load(weights, device=self.device)
        # 获取模型中卷积层的最大步幅
        stride = int(self.model.stride.max())
        # 这行代码使用 check_img_size 函数检查图像的大小（imgsz 变量），并根据步幅 stride 进行调整。这可能是确保输入图像的尺寸与模型的步幅兼容。
        self.imgsz = check_img_size(imgsz, s=stride)
        # 根据需要将模型的精度设置为半精度。
        if self.half:
            self.model.half()

        # get names and colors
        self.names = self.model.module.names if hasattr(
            self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255)
                        for _ in range(3)] for _ in self.names]
        print("model initaial done")

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
                img = cv2.cvtColor(numpy_image, cv2.COLOR_GRAY2RGB)
                if img is not None:
                    showimg = img
                    with torch.no_grad():
                        img = letterbox(img, new_shape=self.opt.img_size)[0]
                        img = img[:, :, ::-1].transpose(2, 0, 1)
                        img = np.ascontiguousarray(img)
                        img = torch.from_numpy(img).to(self.device)
                        img = img.half() if self.half else img.float()
                        img /= 255.0
                        if img.ndimension() == 3:
                            img = img.unsqueeze(0)
                        pred = self.model(img, augment=self.opt.augment)[0]
                        pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres,
                                                   classes=self.opt.classes, agnostic=self.opt.agnostic_nms)

                        accumulated_counts = 0

                        for j, det in enumerate(pred):
                            if det is not None and len(det):
                                det[:, :4] = scale_coords(
                                    img.shape[2:], det[:, :4], showimg.shape).round()

                                for *xyxy, conf, cls in reversed(det):

                                    accumulated_counts +=1

                                    label = '%s %.2f' % (self.names[int(cls)], conf)
                                    name_list.append(self.names[int(cls)])
                                    plot_one_box(
                                        xyxy, showimg, label=label, color=self.colors[int(cls)], line_thickness=2)

                        # show = cv2.resize(showimg, (640, 480))
                        result = cv2.cvtColor(showimg, cv2.COLOR_BGR2RGB)
                        show_image = QImage(result.data, result.shape[1], result.shape[0], QImage.Format_RGB888)
                        print(accumulated_counts)
                        if i == 0:
                            Cam1width=self.Cam1.width()
                            aspect_ratio = show_image.height() / show_image.width()
                            new_height = int(Cam1width * aspect_ratio)
                            scaledPixmap = show_image.scaled(Cam1width, new_height, Qt.KeepAspectRatio,
                                                         Qt.SmoothTransformation)
                            self.Cam1.setScaledContents(False)
                            self.Cam1.setPixmap(QPixmap.fromImage(scaledPixmap))
                        elif i == 1:
                            Cam2width = self.Cam2.width()
                            aspect_ratio = show_image.height() / show_image.width()
                            new_height = int(Cam2width * aspect_ratio)
                            scaledPixmap = show_image.scaled(Cam1width, new_height, Qt.KeepAspectRatio,
                                                             Qt.SmoothTransformation)
                            self.Cam2.setScaledContents(False)
                            self.Cam2.setPixmap(QPixmap.fromImage(scaledPixmap))
                        elif i == 2:
                            Cam3width = self.Cam3.width()
                            aspect_ratio = show_image.height() / show_image.width()
                            new_height = int(Cam3width * aspect_ratio)
                            scaledPixmap = show_image.scaled(Cam1width, new_height, Qt.KeepAspectRatio,
                                                             Qt.SmoothTransformation)
                            self.Cam3.setScaledContents(False)
                            self.Cam3.setPixmap(QPixmap.fromImage(scaledPixmap))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())
