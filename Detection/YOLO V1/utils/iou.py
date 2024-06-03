"""
实现几种不同类型的 iou 计算方法，此处实现了 iou、g_iou、 d_iou、 c_iou
"""

import math
import tensorflow as tf


class IOU:
    def __init__(self):
        """
        各种不同类型的 iou 算法，输入的参数为坐标列表，应分别为 x_min, y_min, x_max, y_max
        """
        pass

    @staticmethod
    def iou(true_box, prediction_box):
        """
        通用 iou 计算，假定真实框和预测框必定相交且交集和并集的面积不相等
        :param true_box: 真实框左上点在图像坐标系中的坐标及右下点在图像坐标系中的坐标
        :param prediction_box: 预测框左上点在图像坐标系中的坐标及右下点在图像坐标系中的坐标
        :return: 真实框和预测框交并比
        """
        intersession_rectangle_x_left_up = tf.maximum(true_box[..., 0], prediction_box[..., 0])
        intersession_rectangle_y_left_up = tf.maximum(true_box[..., 1], prediction_box[..., 1])
        intersession_rectangle_x_right_down = tf.minimum(true_box[..., 2], prediction_box[..., 2])
        intersession_rectangle_y_right_down = tf.minimum(true_box[..., 3], prediction_box[..., 3])
        intersession_area = tf.maximum(intersession_rectangle_y_right_down - intersession_rectangle_y_left_up, 0) * tf.maximum(intersession_rectangle_x_right_down - intersession_rectangle_x_left_up, 0)
        union_area = (true_box[..., 2] - true_box[..., 0]+1) * (true_box[..., 3] - true_box[..., 1]+1) + (prediction_box[..., 2] - prediction_box[..., 0]+1) * (prediction_box[..., 3] - prediction_box[..., 1]+1) - intersession_area
        intersession_union_ration = intersession_area / union_area + 1e-20
        intersession_union_ration = tf.clip_by_value(intersession_union_ration, clip_value_min=0, clip_value_max=1)
        return intersession_union_ration

    @staticmethod
    def g_iou(true_box, prediction_box):
        """
        当预测框和真实框不相交时IoU恒为0，无法反应两个框之间距离得远近,因此引入 g_iou
        计算公式为： iou - 差集/最小外接矩形；差集指最小外接矩形中除去预测框和真实框的区域
        :param true_box: 真实框左上点在图像坐标系中的坐标及右下点在图像坐标系中的坐标
        :param prediction_box: 预测框左上点在图像坐标系中的坐标及右下点在图像坐标系中的坐标
        :return: 使用 g-iou 算法得出的真实框和预测框的交并比
        """
        # 计算两个框的交集、并集、交并比
        intersession_rectangle_width = tf.maximum(tf.minimum(true_box[..., 2], prediction_box[2]) - tf.maximum(true_box[..., 0], prediction_box[0]), 0)
        intersession_rectangle_height = tf.maximum(tf.minimum(true_box[..., 3], prediction_box[3]) - tf.maximum(true_box[..., 1], prediction_box[1]), 0)
        rectangle_intersession = intersession_rectangle_height * intersession_rectangle_width
        rectangle_union = ((true_box[..., 2] - true_box[..., 0]) * (true_box[..., 3] - true_box[..., 1]) + (prediction_box[..., 2] - prediction_box[..., 0]) * (prediction_box[..., 3] - prediction_box[..., 1]) - rectangle_intersession)
        intersession_union_ration = rectangle_intersession / rectangle_union
        # 计算两个框的最小外接矩形面积
        minimum_bounding_rectangle_width = tf.maximum(tf.maximum(true_box[..., 2], prediction_box[2]) - tf.minimum(true_box[..., 0], prediction_box[..., 0]), 0)
        minimum_bounding_rectangle_height = tf.maximum(tf.maximum(true_box[..., 3], prediction_box[..., 3]) - tf.maximum(true_box[..., 1], prediction_box[..., 1]), 0)
        minimum_bounding_rectangle_area = minimum_bounding_rectangle_height * minimum_bounding_rectangle_width
        # 计算差集
        different_set_area = minimum_bounding_rectangle_area - rectangle_union

        g_iou = intersession_union_ration - different_set_area / minimum_bounding_rectangle_area

        return g_iou

    @staticmethod
    def d_iou(true_box, prediction_box):
        """
        当预测框在真实框内部且预测框面积相等时，GIoU无法区分预测框和真实框的相对位置关系，因此引入 d_iou，计算公式为： iou - (d*d) /(c*c), d指预测框和真实框的中心点距离，c指最小外接矩形的对角线距离
        :param true_box: 真实框左上点在图像坐标系中的坐标及右下点在图像坐标系中的坐标
        :param prediction_box: 预测框左上点在图像坐标系中的坐标及右下点在图像坐标系中的坐标
        :return: 使用 d_iou 得到的真实框和预测框的交并比
        """
        # 计算两个框的交集、并集、交并比
        intersession_rectangle_width = tf.maximum(tf.minimum(true_box[..., 2], prediction_box[..., 2]) - tf.maximum(true_box[..., 0], prediction_box[..., 0]), 0)
        intersession_rectangle_height = tf.maximum(tf.minimum(true_box[..., 3], prediction_box[..., 3]) - tf.maximum(true_box[..., 1], prediction_box[..., 1]), 0)
        rectangle_intersession = intersession_rectangle_height * intersession_rectangle_width
        rectangle_union = ((true_box[..., 2] - true_box[..., 0]) * (true_box[..., 3] - true_box[..., 1]) + (prediction_box[..., 2] - prediction_box[..., 0]) * (prediction_box[..., 3] - prediction_box[..., 1]) - rectangle_intersession)
        intersession_union_ration = rectangle_intersession / rectangle_union
        # 计算预测框和真实框的中心点距离 d 的平方
        center_rectangle_1_x = (true_box[..., 2] - true_box[..., 0]) / 2
        center_rectangle_1_y = (true_box[..., 3] - true_box[..., 1]) / 2
        center_rectangle_2_x = (prediction_box[..., 2] - prediction_box[..., 0]) / 2
        center_rectangle_2_y = (prediction_box[..., 3] - prediction_box[..., 1]) / 2
        pow_2_d = tf.square((center_rectangle_2_y - center_rectangle_1_y)) + tf.square((center_rectangle_2_x - center_rectangle_1_x))
        # 计算最小外接矩形坐标点
        minimum_bounding_rectangle_y_max = tf.maximum(true_box[..., 3], prediction_box[..., 3])
        minimum_bounding_rectangle_y_min = tf.minimum(true_box[..., 1], prediction_box[..., 1])
        minimum_bounding_rectangle_x_max = tf.maximum(true_box[..., 2], prediction_box[..., 2])
        minimum_bounding_rectangle_x_min = tf.minimum(true_box[..., 0], prediction_box[..., 0])
        # 计算最小外接矩形对角线距离 c 的平方
        pow_2_c = tf.square(minimum_bounding_rectangle_x_max - minimum_bounding_rectangle_x_min) + tf.square(minimum_bounding_rectangle_y_max - minimum_bounding_rectangle_y_min)

        d_iou = intersession_union_ration - pow_2_d / pow_2_c

        return d_iou

    @staticmethod
    def c_iou(true_box, prediction_box):
        """
        使用 d_iou 时，当预测框在真实框内部且预测框面积以及中心距离相等时，g_iou 无法区分预测框和真实框的长宽比关系,引入 c_iou，计算公式 c_iou = iou - d^2/c^2 - v^2/((1-iou)+v) 其中d dd为预测框和真实框中心点的距离，c cc为最小外接矩形的对角线距离，ν \nuν是长宽比的相似性因子，
        v = (4/pi^2)(arc_tan(Wb/Hb)-arc_tan(Wp/Hp))^2,Wb、Hb、Wp、Hp 分别为真实框宽、高以及预测框宽、高
        :param true_box: 真实框左上点在图像坐标系中的坐标及右下点在图像坐标系中的坐标
        :param prediction_box: 预测框左上点在图像坐标系中的坐标及右下点在图像坐标系中的坐标
        :return: 使用 c_iou 算法计算出的真实框和预测框的交并比
        """
        # 计算两个框的交集、并集、交并比
        intersession_rectangle_width = tf.maximum(tf.minimum(true_box[..., 2], prediction_box[..., 2]) - tf.maximum(true_box[..., 0], prediction_box[..., 0]), 0)
        intersession_rectangle_height = tf.maximum(tf.minimum(true_box[..., 3], prediction_box[..., 3]) - tf.maximum(true_box[..., 1], prediction_box[..., 1]), 0)
        rectangle_intersession = intersession_rectangle_height * intersession_rectangle_width
        rectangle_union = ((true_box[..., 2] - true_box[..., 0]) * (true_box[..., 3] - true_box[..., 1]) + (prediction_box[..., 2] - prediction_box[..., 0]) * (prediction_box[..., 3] - prediction_box[..., 1]) - rectangle_intersession)
        intersession_union_ration = rectangle_intersession / rectangle_union
        # 计算预测框和真实框的中心点距离 d 的平方
        center_rectangle_1_x = (true_box[..., 2] - true_box[..., 0]) / 2
        center_rectangle_1_y = (true_box[..., 3] - true_box[..., 1]) / 2
        center_rectangle_2_x = (prediction_box[..., 2] - prediction_box[..., 0]) / 2
        center_rectangle_2_y = (prediction_box[..., 3] - prediction_box[..., 1]) / 2
        pow_2_d = tf.square(center_rectangle_2_y - center_rectangle_1_y) + tf.square(center_rectangle_2_x - center_rectangle_1_x)
        # 计算最小外接矩形坐标点
        minimum_bounding_rectangle_y_max = tf.maximum(true_box[..., 3], prediction_box[..., 3])
        minimum_bounding_rectangle_y_min = tf.minimum(true_box[..., 1], prediction_box[..., 1])
        minimum_bounding_rectangle_x_max = tf.maximum(true_box[..., 2], prediction_box[..., 2])
        minimum_bounding_rectangle_x_min = tf.minimum(true_box[..., 0], prediction_box[..., 0])
        # 计算最小外接矩形对角线距离 c 的平方
        pow_2_c = tf.square(minimum_bounding_rectangle_x_max - minimum_bounding_rectangle_x_min) + tf.square(minimum_bounding_rectangle_y_max - minimum_bounding_rectangle_y_min)
        # 计算长宽比的相似因子
        v = (4 / tf.square(math.pi)) * tf.square(math.atan((true_box[..., 2] - true_box[..., 0]) / (true_box[..., 3], true_box[..., 1])) - math.atan((prediction_box[..., 2] - prediction_box[..., 0]) / (prediction_box[..., 3], prediction_box[..., 1])))
        c_iou = intersession_union_ration - pow_2_d / pow_2_c - tf.square(v) / (1 - intersession_union_ration + v)

        return c_iou
