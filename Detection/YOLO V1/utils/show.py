import os
import random
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as eT
from utils.config import object_class


class Show:
    def __init__(self):
        pass

    @staticmethod
    def show_train_curve(history):
        """
        绘出模型训练过程中损失函数和准确率的变化过程
        :return:
        """
        loss = history.history["loss"]
        val_loss = history.history["val_loss"]
        plt.ion()
        plt.figure()
        plt.plot(range(1, len(loss) + 1), loss, label="Training Loss")
        plt.plot(range(1, len(val_loss) + 1), val_loss, label="Validation Loss")
        plt.legend(loc="upper right")
        plt.title("Training and Validation Loss Curve")
        plt.show(block=False)
        plt.pause(200)
        plt.close("all")

    @staticmethod
    def show_image_ground_truth_result(file_path, image_shape):
        """
        从数据集中选择四张图像及其标注框进行展示
        :param file_path: 样本文件路径
        :param image_shape: 图像大小
        :return: None
        """
        show_images = []
        if len(os.listdir(os.path.join(file_path, "image"))) <= 4:
            to_show_images = os.listdir(file_path)
        else:
            to_show_images = random.sample(os.listdir(os.path.join(file_path, "image")), k=4)
        for image in to_show_images:
            image_name = os.path.join(file_path, "image", image)
            label_name = os.path.join(file_path, "xml", (image.split(".")[0] + ".xml"))
            image = cv.imread(image_name)
            tree = eT.parse(label_name)
            root = tree.getroot()
            object_list = root.findall("object")
            for object_ in object_list:
                object_name = object_.find("name").text
                object_bn_box = object_.find("bndbox")
                x_min = object_bn_box.find('xmin').text
                y_min = object_bn_box.find('ymin').text
                x_max = object_bn_box.find('xmax').text
                y_max = object_bn_box.find('ymax').text
                cv.putText(image, text=object_name, fontScale=0.5, color=(0, 0, 255), fontFace=0, org=(int(x_min), int(y_min) - 4))
                cv.rectangle(image, pt1=(int(x_min), int(y_min)), pt2=(int(x_max), int(y_max)), lineType=0, color=(0, 255, 0))
            image = cv.resize(src=image, dsize=(image_shape[0], image_shape[1]))
            show_images.append(image)
        horizontal_images_one = np.hstack((show_images[0], show_images[1]))
        horizontal_images_two = np.hstack((show_images[2], show_images[3]))
        vertical_images = np.vstack((horizontal_images_one, horizontal_images_two))
        cv.namedWindow(winname="image_ground_box_display", flags=cv.WINDOW_AUTOSIZE)
        cv.imshow(winname="image_ground_box_display", mat=vertical_images)
        cv.waitKey(0)
        cv.destroyWindow(winname="image_ground_box_display")

    @staticmethod
    def show_predict_image_result(image, detection_result):
        """
        对预测结果进行展示
        :param image: 要绘制预测结果的图像
        :param detection_result: 加载模型的预测结果
        :return: None
        """
        class_id_name = dict(zip(range(0, len(object_class)), object_class))
        image = cv.imread(image)
        for predict_box_result in detection_result:
            # predict_box_result = result[:4].numpy()
            # predict_confidence_result = result[4].numpy()
            # predict_class_result = result[5:].numpy()
            # predict_class_result = np.argmax(predict_class_result)
            # predict_class_result = class_id_name[predict_class_result]
            # cv.putText(image, text=predict_class_result, fontScale=0.5, color=(0, 0, 255), fontFace=0, org=(int(100), int(100) - 4))
            # cv.putText(image, text=str(predict_confidence_result), fontScale=0.5, color=(0, 0, 255), fontFace=0, org=(int(150), int(150) - 4))
            cv.rectangle(image, pt1=(int(predict_box_result[0] ), int(predict_box_result[0])), pt2=(int(predict_box_result[1]), int(predict_box_result[1])), lineType=0, color=(0, 255, 0))
        cv.namedWindow(winname="image_prediction_result", flags=cv.WINDOW_AUTOSIZE)
        cv.imshow(winname="image_prediction_result", mat=image)
        cv.waitKey(0)
        cv.destroyWindow(winname="image_prediction_result")

    @staticmethod
    def show_xx():
        pass
