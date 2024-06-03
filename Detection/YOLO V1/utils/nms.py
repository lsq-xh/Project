"""
nms 作为一种目标检测后处理方法，效果较好的 nms 算法能够有效提高 ap/map 指标， 是目标检测的一个优化方向
实现了3种不同类别的 nms 算法： 传统 nms(hard-nms)、 soft-nms、 d-iou-nms, 后续补充 iou_guided_nms、Weighted_nms、Softer_nms 、Adaptive_nms
分类优先：传统NMS，Soft-NMS   done
定位优先：IoU-Guided NMS     done
加权平均：Weighted NMS
方差加权平均：Softer-NMS
自适应阈值：Adaptive NMS
+中心点距离：DIoU-NMS   done
"""

import numpy as np
import tensorflow as tf


class NMS:
    def __init__(self, confidence_thresh_hold, iou_thresh_hold):
        """
        各种不同类型的nm算法
        :param iou_thresh_hold: 交并比的阈值
        :param confidence_thresh_hold: 置信度的阈值
        """
        self.confidence_thresh = confidence_thresh_hold
        self.iou_thresh = iou_thresh_hold

    @staticmethod
    def iou(true_box, prediction_box):
        """
        通用 iou 计算，假定真实框和预测框必定相交且交集和并集的面积不相等
        :param true_box: 真实框左上点在图像坐标系中的坐标及右下点在图像坐标系中的坐标
        :param prediction_box: 预测框左上点在图像坐标系中的坐标及右下点在图像坐标系中的坐标
        :return: 真实框和预测框交并比
        """
        intersession_rectangle_x_left_up = tf.maximum(true_box[0], prediction_box[0])
        intersession_rectangle_y_left_up = tf.maximum(true_box[1], prediction_box[1])
        intersession_rectangle_x_right_down = tf.minimum(true_box[2], prediction_box[2])
        intersession_rectangle_y_right_down = tf.minimum(true_box[3], prediction_box[3])
        intersession_area = tf.maximum(intersession_rectangle_y_right_down - intersession_rectangle_y_left_up, 0) * tf.maximum(intersession_rectangle_x_right_down - intersession_rectangle_x_left_up, 0)
        union_area = (true_box[2] - true_box[0] + 1) * (true_box[3] - true_box[1] + 1) + (prediction_box[2] - prediction_box[0] + 1) * (prediction_box[3] - prediction_box[1] + 1) - intersession_area
        intersession_union_ration = intersession_area / union_area + 1e-20
        intersession_union_ration = tf.clip_by_value(intersession_union_ration, clip_value_min=0, clip_value_max=1)

        return intersession_union_ration

    def nms(self, prediction_bn_box):
        """
        对检测生成的 bn_box 进行nms算法，获得可能性最大的检测框，prediction_box格式应为：[[x_min, y_min, x_max, y_max, class_confidence, class], .... , [x_min, y_min, x_max, y_max, class_confidence, class]]
        剔除机制严格，对重叠较多的物体容易直接剔除置信度较低的哪个框；根据经验选取阈值；
        :return: 最后在图像中保留的预测框
        """
        after_nms_box = []
        # 根据置信度对预测框列表进行排序
        prediction_box_sorted_based_confidence = sorted(prediction_bn_box, reverse=True, key=lambda score: score[4])
        # 删除置信度低于置信度阈值的预测框
        # print(np.array(prediction_box_sorted_based_confidence).shape)
        # print(len(prediction_box_sorted_based_confidence))
        # prediction_box_sorted_based_confidence = [box for box in prediction_box_sorted_based_confidence if box[4] >= self.confidence_thresh]
        # print(len(prediction_box_sorted_based_confidence))
        # for index in range(0, len(prediction_box_sorted_based_confidence)):
        #     print(prediction_box_sorted_based_confidence[index][4])

        # 取出置信度最高地预测框并放入一个空列表中，同时从原列表中删除此预测框，用此预测框与原列表中的预测框分别计算 iou 删除高于阈值的保留低于阈值的，循环此动作直至原列表中无预测框。
        while len(prediction_box_sorted_based_confidence) > 0:
            most_possible_box = prediction_box_sorted_based_confidence[0]
            prediction_box_sorted_based_confidence.pop(0)
            after_nms_box.append(most_possible_box)
            for box in prediction_box_sorted_based_confidence:
                # 可根据需求替换为不同的 iou 计算方式, most_possible_box[4] == box[4] 加入一个判断是否为同种类别物体，减少运算量
                if most_possible_box[4] == box[4]:
                    iou = self.iou(box[:4], most_possible_box[:4])
                    if iou > self.iou_thresh:
                        prediction_box_sorted_based_confidence.remove(box)

            return after_nms_box

    def soft_nms(self, sigma, mode, nms_thresh, prediction_bn_box):
        """
        对检测生成的 bn_box 进行nms算法，获得可能性最大的检测框，prediction_box格式应为：[[x_min, y_min, x_max, y_max, class_confidence, class], .... , [x_min, y_min, x_max, y_max, class_confidence, class]]
        对 nms 算法进行改进，解决相同物体间预测框重复度较高的问题，可采用高斯法和线性法两种方式来对物体置信度进行修改
        存在定位与得分不一致的情况，则可能导致定位好而得分低的框比定位差得分高的框惩罚更多(遮挡情况下)
        :return: 最后在图像中保留的预测框
        """
        after_nms_box = []
        # 根据置信度对预测框列表进行排序
        prediction_box_sorted_based_confidence = sorted(prediction_bn_box, reverse=True, key=lambda score: score[4])
        # 删除置信度低于置信度阈值的预测框
        for index in range(len(prediction_box_sorted_based_confidence)):
            if prediction_box_sorted_based_confidence[index][4] < self.confidence_thresh:
                prediction_box_sorted_based_confidence.remove(prediction_box_sorted_based_confidence[index])
        # 取出置信度最高地预测框并放入一个空列表中，同时从原列表中删除此预测框，用此预测框与原列表中的预测框分别计算 iou 删除高于阈值的保留低于阈值的，循环此动作直至原列表中无预测框。
        while len(prediction_box_sorted_based_confidence) > 0:
            most_possible_box = prediction_box_sorted_based_confidence[0]
            prediction_box_sorted_based_confidence.pop(0)
            after_nms_box.append(most_possible_box)
            for box in prediction_box_sorted_based_confidence:
                # 可根据需求替换为不同的 iou 计算方式, most_possible_box[4] == box[4] 加入一个判断是否为同种类别物体，减少运算量
                if most_possible_box[4] == box[4]:
                    iou_value = self.iou(box[:4], most_possible_box[:4])
                    if mode == 1:
                        box[4] = box[4] * (1 - int(iou_value))
                    elif mode == 2:
                        box[4] = box[4] * np.exp(-(pow(iou_value, 2) / sigma))
                    else:
                        print("please enter the right mode, it can be 1 or 2 ,1 means using gaussian method, and 2 means liner method")
                        pass
            for box in prediction_box_sorted_based_confidence:
                if box[4] < nms_thresh:
                    prediction_box_sorted_based_confidence.remove(box)
            prediction_box_sorted_based_confidence = sorted(prediction_bn_box, reverse=True, key=lambda score: score[4])

        after_nms_box = prediction_box_sorted_based_confidence

        return after_nms_box

    def d_iou_nms(self, prediction_bn_box):
        """
        对检测生成的 bn_box 进行 d_iou_nms算法，获得可能性最大的检测框，bn_box格式应为：[[x_min, y_min, x_max, y_max, class_confidence, class], .... , [x_min, y_min, x_max, y_max, class_confidence, class]]
        具有和 d_iou 相同的优点、以 d_iou 作为评判指标，能够更好的反应真实框和预测框之间关系
        :return: 最后在图像中保留的预测框
        """
        after_d_iou_nms_box = []
        # 根据置信度对预测框列表进行排序
        bn_box_sorted_based_confidence = sorted(prediction_bn_box, reverse=True, key=lambda score: score[4])
        # 删除置信度低于置信度阈值的预测框
        for box in bn_box_sorted_based_confidence:
            if box[5] < self.confidence_thresh:
                bn_box_sorted_based_confidence.remove(box)
        # 取出置信度最高地预测框并放入一个空列表中，同时从原列表中删除此预测框，用此预测框与原列表中的预测框分别计算 iou 删除高于阈值的保留低于阈值的，循环此动作直至原列表中无预测框。
        while len(bn_box_sorted_based_confidence) > 0:
            most_possible_box = bn_box_sorted_based_confidence[0]
            bn_box_sorted_based_confidence.pop(0)
            after_d_iou_nms_box.append(most_possible_box)
            for box in bn_box_sorted_based_confidence:
                # 可根据需求替换为不同的 iou 计算方式, most_possible_box[4] == box[4] 加入一个判断是否为同种类别物体，减少运算量
                if most_possible_box[4] == box[4]:
                    iou = self.iou(box[:4], most_possible_box[:4]).d_iou()
                    if iou > self.iou_thresh:
                        bn_box_sorted_based_confidence.remove(box)

            return after_d_iou_nms_box
