"""
说明：
    使用tf.data实现数据集的生成。
"""
import os
import random
import numpy as np
import tensorflow as tf
from tensorflow import data
from data.parse import FileParse
from data.augment import DataAugment


class DataLoader:
    def __init__(self, file_base_dir, batch_size, list_of_class_name_of_object):
        """
        初始化函数,初始化文件路径、图像大小、训练时的batch
        :param file_base_dir: 数据集存放路径
        :param batch_size:  输入网络图像的batch_size
        :param list_of_class_name_of_object: 数据集中目标的种类列表，[class_1, class_2, ....class_n]
        """
        self.file_base_dir = file_base_dir
        self.image_path = os.path.join(self.file_base_dir, "JPEGImages")
        self.annotation_path = os.path.join(self.file_base_dir, "Annotations")
        self.batch_size = batch_size
        self.list_of_class_name_of_object = list_of_class_name_of_object

    def get_image_dataset_from_image_set_direct(self, train_val_test_ration):
        """
        读取所有图片数据并根据需要按比例划分数据集
        :param train_val_test_ration: 数据集划分比例，为列表形式
        :return: 划分后的数据集，返回训练集、验证集、测试集，数据集格式：[[image_1,label_1],....[image_n,label_n]]
        """
        image_label_dataset = []
        train_val_test_ds_image = os.listdir(self.image_path)
        for image_name in train_val_test_ds_image:
            label_name = image_name.split(".")[0] + ".xml"
            image_label_dataset.append([os.path.join(self.image_path, image_name), os.path.join(self.annotation_path, label_name)])
        random.shuffle(image_label_dataset)
        train_dataset = image_label_dataset[:int(len(image_label_dataset) * train_val_test_ration[0])]
        val_dataset = image_label_dataset[int(len(image_label_dataset) * train_val_test_ration[0]):int(len(image_label_dataset) * (train_val_test_ration[0] + train_val_test_ration[1]))]
        test_dataset = image_label_dataset[int(len(image_label_dataset) * (train_val_test_ration[0] + train_val_test_ration[1])):]
        return train_dataset, val_dataset, test_dataset

    def get_image_dataset_from_image_set_file(self, train_val_test_ration):
        if not os.path.getsize(os.path.join(self.file_base_dir, "train_val_test.txt")):
            image_files = os.listdir(self.image_path)
            with open(os.path.join(self.file_base_dir, "train_val_test.txt"), "w") as f:
                for index in range(len(image_files)):
                    f.write(image_files[index] + "\\n")
        with open(os.path.join(self.file_base_dir, "train_val_test.txt"), "r") as f:
            image_data = f.readlines()
            for image_data in image_data:
                image_data.strip("\\n")
            random.shuffle(image_data)
            image_label_dataset = []
            for image_name in image_data:
                label_name = image_data.split(".")[0] + ".xml"
                image_label_dataset.append([os.path.join(self.image_path, image_name), os.path.join(self.annotation_path, label_name)])
            train_dataset = image_label_dataset[:int(len(image_label_dataset) * train_val_test_ration[0])]
            val_dataset = image_label_dataset[int(len(image_label_dataset) * train_val_test_ration[0]):int(len(image_label_dataset) * (train_val_test_ration[0] + train_val_test_ration[1]))]
            test_dataset = image_label_dataset[int(len(image_label_dataset) * (train_val_test_ration[0] + train_val_test_ration[1])):]
        return train_dataset, val_dataset, test_dataset

    def get_number_of_images_in_dataset(self, train_val_test_ration, style):
        """
        得到训练集、验证集、测试集中图像的数量
        :param train_val_test_ration: 数据集划分比例，为列表形式
        :param style: 根据不同方式创建的数据的方式选择
        :return:训练集、验证集、测试集中图像的数量
        """
        if style == 0:
            train_dataset, val_dataset, test_dataset = self.get_image_dataset_from_image_set_direct(train_val_test_ration)
        else:
            train_dataset, val_dataset, test_dataset = self.get_image_dataset_from_image_set_file(train_val_test_ration)
        return len(train_dataset), len(val_dataset), len(test_dataset)

    def data_generator(self, datasets, target_image_shape):
        """
        读取一张图片并解析其标签文件，生成图像和标签
        :param datasets: 训练集/验证集/测试集
        :param target_image_shape:要生成的训练图像的形状
        :return: 不断生成的图像及其标签
        """

        # 仅 yolo_v1 需要在此处设置此变量,因为yolo v1 最后输出向量长度位： 7x7x(2*5+20)
        grid_size = [7, 7]
        # 仅 yolo_v1 需要在此处设置此变量

        for image_label in datasets:
            image_name = image_label[0].decode("utf-8")
            label_name = image_label[1].decode("utf-8")
            image = tf.io.read_file(image_name)
            image = tf.io.decode_image(image, channels=3)
            image = tf.image.resize(image, size=[target_image_shape[0], target_image_shape[1]])
            image = image / 255.
            label = np.zeros(shape=[*grid_size, 5 + len(self.list_of_class_name_of_object)])
            parsed_label = FileParse(target_image_shape=target_image_shape, list_of_class_name_of_object=self.list_of_class_name_of_object).call_parse_file(flag="xml", file_name=label_name)
            for label_item in parsed_label:
                grid_coordinate_x = int(label_item[1] * grid_size[0] / image.shape[0])
                grid_coordinate_y = int(label_item[2] * grid_size[1] / image.shape[1])
                label[grid_coordinate_x, grid_coordinate_y, 0] = 1
                label[grid_coordinate_x, grid_coordinate_y, 1:] = label_item[:]

            yield image, label

    def get_train_dataset(self, image_shape, datasets):
        """
        生成tf.data格式的训练集
        :param image_shape:要生成的训练图像的形状
        :param datasets:数据集
        :return: tf.data的训练集
        """
        train_image_label_data = data.Dataset.from_generator(generator=self.data_generator, args=[datasets, image_shape], output_types=(tf.float32, tf.float32),
                                                             output_shapes=([image_shape[0], image_shape[1], image_shape[2]], None))  # 5 + len(bin(self.object_classes_number))
        # train_image_label_data = train_image_label_data.map(DataAugment(image_shape=image_shape).wrapped_image_augment, num_parallel_calls=tf.data.AUTOTUNE)
        train_image_label_data = train_image_label_data.batch(self.batch_size, drop_remainder=False, num_parallel_calls=tf.data.AUTOTUNE)
        train_image_label_data = train_image_label_data.prefetch(buffer_size=tf.data.AUTOTUNE)
        return train_image_label_data

    def get_val_dataset(self, image_shape, datasets):
        """
        生成tf.data格式的训练集
        :param image_shape:要生成的训练图像的形状
        :param datasets:数据集
        :return: tf.data的训练集
        """
        train_image_label_data = data.Dataset.from_generator(generator=self.data_generator, args=[datasets, image_shape], output_types=(tf.float32, tf.float32),
                                                             output_shapes=([image_shape[0], image_shape[1], image_shape[2]], None))
        train_image_label_data = train_image_label_data.batch(self.batch_size, drop_remainder=False, num_parallel_calls=tf.data.AUTOTUNE)
        train_image_label_data = train_image_label_data.prefetch(buffer_size=tf.data.AUTOTUNE)
        return train_image_label_data

    def get_test_dataset(self, image_shape, datasets):
        """
        生成tf.data格式的训练集
        :param image_shape:要生成的训练图像的形状
        :param datasets:数据集
        :return: tf.data的训练集
        """
        train_image_label_data = data.Dataset.from_generator(generator=self.data_generator, args=[datasets, image_shape], output_types=(tf.float32, tf.float32),
                                                             output_shapes=([image_shape[0], image_shape[1], image_shape[2]], None))
        train_image_label_data = train_image_label_data.batch(self.batch_size, drop_remainder=False, num_parallel_calls=tf.data.AUTOTUNE)
        train_image_label_data = train_image_label_data.prefetch(buffer_size=tf.data.AUTOTUNE)
        return train_image_label_data


