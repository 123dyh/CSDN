import sys
from PyQt5.QtWidgets import QMessageBox, QFileDialog, QLineEdit
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

from models.experimental import attempt_load
from utils.augmentations import letterbox
from untitled2 import Ui_MainWindow
import numpy as np

import cv2
import shutil
import PyQt5.QtCore
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import threading
import argparse
import os
import sys
from pathlib import Path
import cv2
import random
import torch
import torch.backends.cudnn as cudnn
import os.path as osp

from PyQt5 import QtCore, QtGui, QtWidgets


from openpyxl import Workbook
from openpyxl.drawing.image import Image as ExcelImage
from openpyxl.drawing.image import Image as OpenpyxlImage
from pathlib import Path
from datetime import datetime

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box, plot_one_box
from utils.torch_utils import select_device, time_sync
from datetime import datetime


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        # UI界面
        self.setupUi(self)
        self.excel_row = 2
        self.wb = Workbook()  # 初始化工作簿
        self.ws = self.wb.active  # 获取活动的工作表
        self.ws.title = "Defects"
        headers = ['Class', 'Confidence', 'Image']
        self.ws.append(headers)  # 添加标题行
        now = datetime.now()
        formatted_date_time = now.strftime("%Y-%m-%d_%H-%M-%S")
        self.excel_filename = f"defects_info_{formatted_date_time}.xlsx"
        self.timer_video = QtCore.QTimer()
        self.output_size = 480
        self.showFullScreen()
        # self.init_slots()
        # self.CAM_NUM = 0

        self.cap = cv2.VideoCapture("0")
        # self.background()
        self.device = '0'
        self.timer_video.timeout.connect(self.show_video_frame)
        self.model1=self.model_init()
        self.camera_open()


        self.save_path = 'D:/333'  # 指定保存图片的路径
        self.det_count = 1 # 用于图片命名的起始数字
        self.accumulated_counts=0
        self.tableWidget.cellClicked.connect(self.cell_was_clicked)

    def model_init(self):
        # 创建了一个解析器对象。
        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', nargs='+', type=str, default='D:/Python/1.Python/Pycharm2021/yolov5-master/video_check/ruanjian/ruanjian/best(2).pt',
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


    def show_video_frame(self):
        name_list = []
        flag, img = self.cap.read()

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
                # Inference
                pred = self.model(img, augment=self.opt.augment)[0]

                # Apply NMS
                pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, classes=self.opt.classes,
                                           agnostic=self.opt.agnostic_nms)

                # Process detections
                for det in pred:  # detections per image
                    if det is not None and len(det):
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], showimg.shape).round()

                        for *xyxy, conf, cls in reversed(det):
                            label = f'{self.names[int(cls)]} {conf:.2f}'
                            name_list.append(self.names[int(cls)])
                            plot_one_box(xyxy, showimg, label=label, color=self.colors[int(cls)], line_thickness=2)

                            image_path = f"{self.save_path}/{self.det_count}.jpg"
                            resized_img = cv2.resize(showimg, (280, 270), interpolation=cv2.INTER_LINEAR)
                            cv2.imwrite(image_path, resized_img)

                            class_name = self.names[int(cls)]
                            confidence = conf.item()
                            self.update_table(class_name, conf, image_path)
                            self.ws.append([class_name, confidence, image_path])

                            path='c'
                            self.ws.column_dimensions[path].width=20

                            img_column = 'D'
                            self.ws.row_dimensions[self.excel_row].height = 200
                            self.ws.column_dimensions[img_column].width = 40

                            img_excel = OpenpyxlImage(image_path)
                            img_excel.anchor = f"{img_column}{self.excel_row}"
                            self.ws.add_image(img_excel)

                            self.det_count += 1
                            if self.accumulated_counts>=7:
                                self.accumulated_counts=0
                            else:
                                self.accumulated_counts+=1
                            # print(self.det_count)
                            print(self.accumulated_counts)

                            self.excel_row += 1

                self.wb.save(self.excel_filename)
                self.out.write(showimg)
                show = cv2.resize(showimg, (640, 480))
                self.result = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
                showImage = QtGui.QImage(self.result.data, self.result.shape[1], self.result.shape[0],
                                         QtGui.QImage.Format_RGB888)



                self.Cam1.setPixmap(QtGui.QPixmap.fromImage(showImage))

    def update_table(self, class_name, confidence, image_path):
        row_position = self.tableWidget.rowCount()
        self.tableWidget.insertRow(row_position)
        self.tableWidget.setItem(row_position, 0, QTableWidgetItem(str(self.det_count)))
        self.tableWidget.setItem(row_position, 1, QTableWidgetItem(class_name))
        self.tableWidget.setItem(row_position, 2, QTableWidgetItem(f"{confidence:.2f}"))

        # 为详情列添加一个项目，并设置用户数据为图片路径
        photo_item = QTableWidgetItem("图片")
        photo_item.setData(Qt.UserRole, image_path)
        self.tableWidget.setItem(row_position, 3, photo_item)

    def cell_was_clicked(self, row, column):
        # 检查是否点击的是“详情”列
        if column == 3:
            # 从单元格中获取图片路径
            item = self.tableWidget.item(row, column)
            image_path = item.data(Qt.UserRole)
            if image_path:
                self.show_image(image_path)

    def show_image(self, image_path):
        pixmap = QPixmap(image_path)
        self.show1.setPixmap(pixmap)

    def camera_open(self):
        if not self.timer_video.isActive():
            # 默认使用第一个本地camera
            flag = self.cap.open(0)
            if flag == False:
                QtWidgets.QMessageBox.warning(
                    self, u"Warning", u"打开摄像头失败", buttons=QtWidgets.QMessageBox.Ok,
                    defaultButton=QtWidgets.QMessageBox.Ok)
            else:
                self.out = cv2.VideoWriter('prediction.avi', cv2.VideoWriter_fourcc(
                    *'MJPG'), 20, (int(self.cap.get(3)), int(self.cap.get(4))))
                self.timer_video.start(30)
    # 关闭相机
    def close_camera(self):
        self.cap.release()
        self.pushButton.setEnabled(True)
        self.pushButton_2.setEnabled(False)
        self.timer.stop()
if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())
