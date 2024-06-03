""" 计算模型处理图像的帧率 fps"""

import os
import time
import cv2 as cv


class FpsCalculate:
    def __init__(self, model, video_path, images_path):
        """
        初始化参数
        :param model: 用来进行预测的模型
        :param video_path: 视频文件的保存路径
        :param images_path: 图像的保存路径
        """
        self.model = model
        self.video_path = video_path
        self.image_path = images_path

    def calculate_video_fps(self, using_camera_flag):
        """
        计算模型处理视频流的 fps
        :param: using_camera_flag: 是否调用摄像头进行检测
        :return: 模型的 fps
        """
        frame_count = 0
        model = self.model.load_model()
        if using_camera_flag:
            video = cv.VideoCapture(0)
        else:
            video = cv.VideoCapture(self.video_path)
        while True:
            ret, frame = video.read()
            if ret:
                start_time = time.time()
                model.predict(frame)
                frame_count += 1
                current_time = time.time()
                elapsed_time = current_time - start_time
                if elapsed_time >= 1:
                    break
        fps = frame_count / elapsed_time

        return fps

    def calculate_images_fps(self):
        """
        计算处理图像的 fps
        :return: 模型处理图像的 fps
        """
        images_list = os.listdir(self.image_path)
        image_name_list = []
        for image in images_list:
            image_name = os.path.join(self.image_path, image)
            image_name_list.append(image_name)
        model = self.model
        start_time = time.time()
        for image in image_name_list:
            model.predict(image)
        current_time = time.time()
        elapsed_time = current_time - start_time
        fps = len(images_list) / elapsed_time

        return fps
