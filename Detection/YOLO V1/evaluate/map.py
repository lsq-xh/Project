import os
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as eT
from utils.iou import IOU


class MetricsApMap:
    def __init__(self, evaluate_dataset_annotation_dir, evaluate_dataset_image_dir, model, iou_thresh):
        """
        初始化化函数，实现目标检测的评价指标 AP mAP
        :param evaluate_dataset_annotation_dir: 测试数据集注释文件路径
        :param evaluate_dataset_image_dir: 测试数据集图像文件路径
        :param model: 要测试的模型
        :param iou_thresh: 交并比的阈值
        """
        self.evaluate_dataset_annotation_dir = evaluate_dataset_annotation_dir
        self.evaluate_dataset_image_dir = evaluate_dataset_image_dir
        self.model = model
        self.iou_thresh = iou_thresh

    @staticmethod
    def parse_xml_files(xml_file):
        """
        从注释文件中解析出图像的文件名、目标类别、目标真实标注框的坐标
        :param xml_file: 要解析的xml文件
        :return: 要测试数据集中一张图像的所有目标的标注、目标类别、文件名的列表，格式为[[xmin, ymin, xmax, ymax, object_name, image_file_name]....[],[],]
        """
        bn_box_in_one_image = []
        tree = eT.parse(xml_file)
        root = tree.getroot()
        image_file_name = root.find("filename").text
        object_list = root.findall("object")
        for object_ in object_list:
            object_name = object_.find("name").text
            object_bn_box = object_.find("bndbox")
            gt_bn_box = (float(object_bn_box.find('xmin').text), float(object_bn_box.find('ymin').text), float(object_bn_box.find('xmax').text), float(object_bn_box.find('ymax').text))
            bn_box = list(gt_bn_box) + [object_name] + [image_file_name]
            bn_box_in_one_image.append(bn_box)
        return bn_box_in_one_image

    def get_gt_box_of_evaluate_dataset(self):
        """
        获得要测试的数据集中的所有图像的所有目标的信息
        :return: 要测试数据集中所有图像的所有目标的标注框、目标类别、文件名的列表，格式为[[xmin, ymin, xmax, ymax, object_name, image_file_name]....[],[],]
        """
        gt_bn_box_of_all_image = []
        evaluate_dataset_annotation_files = os.listdir(self.evaluate_dataset_annotation_dir)
        evaluate_dataset_annotation_files_dir = [os.path.join(self.evaluate_dataset_annotation_dir, xml_files) for xml_files in evaluate_dataset_annotation_files]
        for evaluate_files in evaluate_dataset_annotation_files_dir:
            gt_bn_box_of_all_image = self.parse_xml_files(evaluate_files)
            gt_bn_box_of_all_image += gt_bn_box_of_all_image
        return gt_bn_box_of_all_image

    def get_prediction_box_of_evaluation_dataset(self):
        """
        获得要测试的数据集中的对所有图像的预测结果
        :return: 测试数据集所有图像的预测得到的目标的标注框、置信度、目标类别，格式为[[xmin, ymin, xmax, ymax, object_name, confidence, image_file_name]....[],[],]
        """
        prediction_bn_box_of_all_image = []
        evaluation_dataset_image_files = os.listdir(self.evaluate_dataset_image_dir)
        evaluation_dataset_image_files_dir = [os.path.join(self.evaluate_dataset_image_dir, jpg_files) for jpg_files in evaluation_dataset_image_files]
        for evaluate_jpg_files in evaluation_dataset_image_files_dir:
            prediction_bn_box_of_all_image = self.model.predict(evaluate_jpg_files) + [evaluate_jpg_files]
            prediction_bn_box_of_all_image += prediction_bn_box_of_all_image
        return prediction_bn_box_of_all_image

    """ 计算 precision 和 recall 的值还需要重点理解如何计算，在进行测试时要重点思考，若正确则删除此行注释-----吕盛强"""

    def calculate_precision_recall(self, object_class):
        """
        计算单一类别物体的 precision 、recall 指标列表
        :return: 一种检测类别的 precision 、 recall 指标列表
        """
        tp = 0
        fp = 0
        precision_recall = []
        prediction_bn_box_of_all_image = self.get_prediction_box_of_evaluation_dataset()
        gt_bn_box_of_all_image = self.get_gt_box_of_evaluate_dataset()
        # 剔除非计算类别的元素框，减少运算量
        for element in prediction_bn_box_of_all_image:
            if element[4] != object_class:
                prediction_bn_box_of_all_image.remove(element)
        for element in gt_bn_box_of_all_image:
            if element[4] != object_class:
                gt_bn_box_of_all_image.remove(element)
        # 计算每一张图片中检测出来的所有单一类别的 precision 和 recall,首先根据置信度对预测框进行降序排列并过滤掉低于某置信度的预测框，然后在计算不同置信度情况下的 precision 和 recall
        prediction_bn_box_of_all_image = sorted(prediction_bn_box_of_all_image, reverse=True, key=lambda score: score[5])
        for confidence_score in prediction_bn_box_of_all_image[:, 5]:
            prediction_bn_box_of_all_image_confidence_filtered = [x for x in prediction_bn_box_of_all_image if x[5] >= confidence_score]
            for element_gt in gt_bn_box_of_all_image:
                for element_predict in prediction_bn_box_of_all_image_confidence_filtered:
                    if element_predict[6] == element_gt[6]:
                        intersession_union_ration = IOU().iou(element_predict[:3], element_gt[:3])
                        if intersession_union_ration > self.iou_thresh:
                            tp = tp + 1
                            fp = fp
                        else:
                            tp = tp
                            fp = fp + 1
            # 根据置信度来计算 tp fp 的值，注意：tp + fn = 测试集数据中所有真实框的个数，因此在计算中也使用测试集中所有真实框的个数来计算 recall 的分母
            precision = tp / (tp + fp)
            recall = tp / len(gt_bn_box_of_all_image)
            precision_recall.append((precision, recall))

        return precision_recall

    def draw_precision_recall_curve(self, object_class):
        """
        根据 precision、recall 指标列表，绘制出一个目标类别的 pr 曲线
        :return:
        """
        precision_recall_list = self.calculate_precision_recall(object_class=object_class)
        precision = []
        recall = []
        for precision_recall in precision_recall_list:
            precision_ = precision_recall[0]
            recall_ = precision_recall[1]
            precision.append(precision_)
            recall.append(recall_)
        plt.figure()
        plt.plot(precision, recall)
        plt.title("precision and recall curve")
        plt.legend(loc="upper right")
        plt.show(block=False)
        plt.pause(20)
        plt.close("all")

    # 还需再了解计算ap的方法，未能理解本质含义，理解后删除此条注释-----吕盛强
    def ap_calculate(self, object_class):
        """
        计算单种类目标类别的 ap 值
        :return: 单种类目标类别的 ap 值
        """
        precision = []
        recall = []
        # 获取 precision 和 recall 对数, 根据 recall 的值对 precision 和recall 数对进行降序排列，通过降序排列后的数据剔除相同 recall 值中 precision 较小的值
        precision_recall = self.calculate_precision_recall(object_class=object_class)
        precision_recall = sorted(precision_recall, reverse=True, key=lambda x: x[1])
        for precision_recall in precision_recall:
            precision_ = precision_recall[0]
            recall_ = precision_recall[1]
            if recall_ not in recall:
                precision.append(precision_)
                recall.append(recall_)
        recall = np.concatenate(([0.], recall, [1.]))
        precision = np.concatenate(([0.], precision, [0.]))
        for index in range(precision.size - 1, 0, -1):
            precision[index - 1] = np.maximum(precision[index - 1], precision[index])
        index_ = np.where(recall[1:] != recall[:-1])[0]
        ap = np.sum((recall[index_ + 1] - recall[index_]) * precision[index_ + 1])

        return ap

    def map_calculate(self, object_class_list):
        """
        计算测试集的 吗AP 值
        :return: 测试集的 吗AP 值
        """
        map_value = 0.
        ap_list = []
        for object_class in object_class_list:
            ap = self.ap_calculate(object_class=object_class)
            ap_list.append(ap)
        for ap in ap_list:
            map_value = map_value + ap

        map_value = map_value / len(object_class_list)

        return map_value
