"""
说明：
    使用 opencv 对数据集进行增强，对图像增强函数进行包装使其能够嵌入 tensorflow 中
"""

import random
import cv2 as cv
import numpy as np
import tensorflow as tf


class DataAugment:
    def __init__(self, image_shape):
        """
        参数初始化
        :param image_shape: 图像的长宽 (width, height)
        """
        self.image_shape = image_shape

    def wrapped_image_augment(self, image, label):
        """
        使用tf.py_function对图像增强函数进行包装,可以使用opencv等外部库对图像进行增强
        :param image: 要进行增强操作的图像
        :param label: 图像增强过程中需要调整的真实框的坐标
        :return: 增强后的图像和标签
        """
        image_augment_method = random.randint(a=0, b=1)
        if image_augment_method == 0:
            image, label = tf.py_function(func=self.image_flip, inp=[image, label], Tout=(tf.float32, tf.float32))
        elif image_augment_method == 1:
            image, label = tf.py_function(func=self.image_blur, inp=[image, label], Tout=(tf.float32, tf.float32))
        else:
            image, label = image, label
        image.set_shape(self.image_shape)
        label = tf.convert_to_tensor(label, dtype=tf.float32)
        return image, label

    def image_flip(self, image, box):
        """
        图像增强算法，对图像进行滤波处理
        :param image: 要增强的图像
        :param box: 要调整的真实框的坐标
        :return: 增强后的图像和调整后的真实框坐标
        """
        image = image.numpy()
        box = box.numpy()
        changed_image = cv.flip(src=image, flipCode=1)
        for box_item in box:
            box_item[2] = 2 * self.image_shape[0] - box_item[2]
        changed_box = box
        return changed_image, changed_box

    @staticmethod
    def image_blur(image, box):
        """
        图像增强算法，对图像进行滤波处理
        :param image: 要增强的图像
        :param box: 要调整的真实框的坐标
        :return: 增强后的图像和调整后的真实框坐标
        """
        image = image.numpy()
        box = box.numpy()
        changed_image = cv.blur(src=image, ksize=(3, 3))
        changed_box = box
        return changed_image, changed_box
