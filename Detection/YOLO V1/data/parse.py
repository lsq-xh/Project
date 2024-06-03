"""
实现对各种格式的标签文件的解析与处理，此次暂时只实现了对 xml 文件的处理
"""
import numpy as np
import xml.etree.ElementTree as eT


class FileParse:
    def __init__(self, target_image_shape, list_of_class_name_of_object):
        """
        初始化函数
         :param target_image_shape：要输入网络的图像的形状 [width, height, channel]
         :param list_of_class_name_of_object: 数据集目标种类名称的列表，[class_name_1, class_name_2 ....]
         :return: None
        """
        self.list_of_class_name_of_object = list_of_class_name_of_object
        self.target_image_shape = target_image_shape

    def encode_object_id(self):
        """
        对目标进行 one_hot 编码
        :return: 编码后的目标物体标签
        """
        id_index = range(0, len(self.list_of_class_name_of_object))
        class_index_encode_map = dict(zip(self.list_of_class_name_of_object, id_index))
        return class_index_encode_map

    def parse_xml_files(self, file_name):
        """
        对以 xml 格式注释的单个文件进行解析
        :param file_name: 要解析的文件
        :return: 解析后得到的标签文件，格式为：[object_class_id, center_x, center_y, width, height], object_class_id 是经过 one_hot 编码后的类别
        """
        bn_box_in_one_image = []
        tree = eT.parse(file_name)
        root = tree.getroot()
        object_list = root.findall("object")
        original_image_shape = root.findall("size")[0]
        original_image_shape_width = original_image_shape.find("width").text
        original_image_shape_height = original_image_shape.find("height").text
        width_resize_scale = float(original_image_shape_width) / float(self.target_image_shape[0])
        height_resize_scale = float(original_image_shape_height) / float(self.target_image_shape[1])
        for object_ in object_list:
            object_name = object_.find("name").text
            encode_object_name = np.eye(len(self.list_of_class_name_of_object), )[np.array(self.encode_object_id().get(object_name), dtype=np.int32)].tolist()
            object_bn_box = object_.find("bndbox")
            resized_x_min = max(min((float(object_bn_box.find('xmin').text) - 1.) * width_resize_scale, float(self.target_image_shape[0]) - 1.), 0)
            resized_y_min = max(min((float(object_bn_box.find('ymin').text) - 1.) * height_resize_scale, float(self.target_image_shape[1]) - 1.), 0)
            resized_x_max = max(min((float(object_bn_box.find('xmax').text) - 1.) * width_resize_scale, float(self.target_image_shape[0]) - 1.), 0)
            resized_y_max = max(min((float(object_bn_box.find('ymax').text) - 1.) * height_resize_scale, float(self.target_image_shape[1]) - 1.), 0)
            resized_gt_box_center_x = (resized_x_min + resized_x_max) / 2
            resized_gt_box_center_y = (resized_y_min + resized_y_max) / 2
            resized_gt_box_width = resized_x_max - resized_x_min
            resized_gt_box_height = resized_y_max - resized_y_min
            resized_gt_bn_box = [resized_gt_box_center_x, resized_gt_box_center_y, resized_gt_box_width, resized_gt_box_height]
            bn_box = resized_gt_bn_box + encode_object_name
            bn_box_in_one_image.append(bn_box)
        return bn_box_in_one_image

    def parse_json_file(self, file_name):
        """
        对以 json 格式注释的单个文件进行解析
        :param file_name: 要解析的文件
        :return: 解析后得到的标签文件，格式为：[object_class, x_min, y_min, x_max, y_max]
        """

        class_index_encode_map = self.encode_object_id()
        pass

    def parse_csv_file(self, file_name):
        """
        对以 csv 格式注释的单个文件进行解析
        :param file_name: 要解析的文件
        :return: 解析后得到的标签文件，格式为：[object_class, x_min, y_min, x_max, y_max]
        """

        class_index_encode_map = self.encode_object_id()
        pass

    def call_parse_file(self, flag, file_name):
        """
        选择要解析的标签文件类型并调用相应的函数进行标签注释解析
        :param flag: 要选择的解析文件类型
        :param file_name: 要解析的文件名
        :return: 解析后得到的标签文件，格式为：[object_class, x_min, y_min, x_max, y_max]
        """
        if flag == "xml":
            labels = self.parse_xml_files(file_name=file_name)
            return labels
        elif flag == "csv":
            pass
            # labels = self.parse_csv_file(file=file_name)
            # return labels
        elif flag == "json":
            pass
            # labels = self.parse_json_file(file=file_name)
            # return labels
        else:
            print("This style label parse is not support at present, it now only can be xml、 csv、 json")
            pass
