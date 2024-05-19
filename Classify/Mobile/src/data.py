"""
说明：
    使用tf.data实现数据集的生成和图像增强。
数据集文件存放格式：
    dir--
       |-- object 1
       |-- object 2
       |-- ........
       |-- object n
"""

import os
import random
import cv2 as cv
import tensorflow as tf
from tensorflow import data


class DataProcess:
    def __init__(self, file_path, image_shape, batch_size):
        """
        初始化函数,初始化文件路径、图像大小、训练时的batch
        :param file_path: 数据集文件存放路径
        :param image_shape: 输入网络的图像形状
        :param batch_size:  输入网络图像的batch_size
        """
        self.image_file_path = file_path
        self.image_shape = image_shape
        self.batch_size = batch_size
        self.train_val_test_ds = self.get_image_dataset()

    def get_object_class_and_number(self):
        """
        获取数据集中需识别对象的种类及每一类目标的图像数量
        :return: 数据集中目标的种类及每一类目标的图像数量
        """
        object_classes_number = len(os.listdir(self.image_file_path))
        object_classes = os.listdir(self.image_file_path)

        return object_classes_number, object_classes

    def get_label_dict(self):
        """
        获得类别和数字编码的对应关系
        :return: 类别和数字编码的对应关系
        """
        object_classes_number, object_classes = self.get_object_class_and_number()
        label_index = dict(zip(object_classes, [object_class_index for object_class_index in range(object_classes_number)]))

        return label_index

    def get_image_dataset(self):
        """
        获取数据集图像，并生成训练集、测试集、验证集。训练集、验证集、测试集的划分比例可在 line54--line56进行修改
        :return: 训练网络所需的训练集、验证集、测试集
        """
        train_dataset = []
        val_dataset = []
        test_dataset = []
        image_classes = self.get_object_class_and_number()[0]
        for classes_index in range(image_classes):
            images_dir = os.path.join(self.image_file_path, os.listdir(self.image_file_path)[classes_index])
            images = os.listdir(images_dir)
            images = [os.path.join(images_dir, image) for image in images]
            images = [image.replace("\\", "/") for image in images]
            image_numbers = len(images)
            train_images = images[:int(image_numbers * 0.6)]
            val_images = images[int(image_numbers * 0.6):int(image_numbers * 0.8)]
            test_images = images[int(image_numbers * 0.8):]
            train_dataset = train_dataset + train_images
            val_dataset = val_dataset + val_images
            test_dataset = test_dataset + test_images
        random.shuffle(train_dataset)
        random.shuffle(val_dataset)
        random.shuffle(test_dataset)

        return train_dataset, val_dataset, test_dataset

    def get_image_number_of_dataset(self):
        """
        获取训练集、测试集、验证集的图像数量
        :return: 训练集、测试集、验证集的图像数量
        """

        train_dataset_image_number = len(self.get_image_dataset()[0])
        val_dataset_image_number = len(self.get_image_dataset()[1])
        test_dataset_image_number = len(self.get_image_dataset()[2])

        return train_dataset_image_number, val_dataset_image_number, test_dataset_image_number

    def data_generator(self, datasets):
        """
        使用生成器生成图像和图像标签
        :param datasets: 训练集/验证集/测试集
        :return: 不断生成图像和标签
        """
        label_index = self.get_label_dict()
        for data_label in datasets:
            data_label = data_label.decode('utf-8')
            image = tf.io.read_file(data_label)
            image = tf.io.decode_image(contents=image, channels=3)
            image = tf.image.resize(images=image, size=(self.image_shape[0], self.image_shape[1]))
            label = label_index[data_label.split("/")[-2]]
            label = tf.one_hot(label, self.get_object_class_and_number()[0])

            yield image, label

    def wrapped_train_image_augment(self, image, label):
        """
        使用tf.py_function对图像增强函数进行包装,可以使用opencv等外部库对图像进行增强
        :param image: 图像
        :param label: 标签
        :return: 增强后的图像和标签
        """
        image, label = tf.py_function(func=self.image_augment, inp=[image, label], Tout=(tf.float32, tf.int32))
        image.set_shape(self.image_shape)
        object_class_numer = self.get_object_class_and_number()[0]
        label.set_shape(object_class_numer)

        return image, label

    def wrapped_val_test_image_augment(self, image, label):
        """
        使用tf.py_function对图像增强函数进行包装,可以使用opencv等外部库对图像进行增强
        :param image: 图像
        :param label: 标签
        :return: 增强后的图像和标签
        """
        image = tf.convert_to_tensor(image, dtype=tf.float32)
        image = image/255.
        label = tf.convert_to_tensor(label, dtype=tf.int32)
        object_class_numer = self.get_object_class_and_number()[0]
        image.set_shape(self.image_shape)
        label.set_shape(object_class_numer)

        return image, label

    def get_train_dataset(self):
        """
        生成tf.data格式的训练集
        :return: tf.data的训练集
        """
        datasets = self.train_val_test_ds[0]
        object_class_numer = self.get_object_class_and_number()[0]
        train_image_label_data = data.Dataset.from_generator(generator=self.data_generator, args=[datasets], output_types=(tf.float32, tf.int32),
                                                             output_shapes=([self.image_shape[0], self.image_shape[1], self.image_shape[2]], [object_class_numer]))
        train_image_label_data = train_image_label_data.map(self.wrapped_train_image_augment, num_parallel_calls=tf.data.AUTOTUNE)
        train_image_label_data = train_image_label_data.batch(self.batch_size, drop_remainder=False, num_parallel_calls=tf.data.AUTOTUNE)
        train_image_label_data = train_image_label_data.prefetch(buffer_size=tf.data.AUTOTUNE)

        return train_image_label_data

    def get_val_dataset(self):
        """
        生成tf.data格式的验证集
        :return: tf.data格式的验证集
        """
        datasets = self.train_val_test_ds[1]
        object_class_numer = self.get_object_class_and_number()[0]
        val_image_label_data = data.Dataset.from_generator(generator=self.data_generator, args=[datasets], output_types=(tf.float32, tf.int32),
                                                           output_shapes=([self.image_shape[0], self.image_shape[1], self.image_shape[2]], [object_class_numer]))
        val_image_label_data = val_image_label_data.map(self.wrapped_val_test_image_augment, num_parallel_calls=tf.data.AUTOTUNE)
        val_image_label_data = val_image_label_data.batch(self.batch_size, drop_remainder=False, num_parallel_calls=tf.data.AUTOTUNE)
        val_image_label_data = val_image_label_data.prefetch(buffer_size=tf.data.AUTOTUNE)

        return val_image_label_data

    def get_test_dataset(self):
        """
        生成tf.data格式的测试集
        :return: tf.data格式的测试集
        """
        datasets = self.train_val_test_ds[2]
        object_class_numer = self.get_object_class_and_number()[0]
        test_image_label_data = data.Dataset.from_generator(generator=self.data_generator, args=[datasets], output_types=(tf.float32, tf.int32),
                                                            output_shapes=([self.image_shape[0], self.image_shape[1], self.image_shape[2]], [object_class_numer]))
        test_image_label_data = test_image_label_data.map(self.wrapped_val_test_image_augment, num_parallel_calls=tf.data.AUTOTUNE)
        test_image_label_data = test_image_label_data.batch(self.batch_size, drop_remainder=False, num_parallel_calls=tf.data.AUTOTUNE)
        test_image_label_data = test_image_label_data.prefetch(buffer_size=tf.data.AUTOTUNE)

        return test_image_label_data

    @staticmethod
    def image_augment(image, label):
        """
        使用opencv库对图像进行随机增强
        :param image: 图像
        :param label: 标签
        :return: 增强后的图像和标签
        """
        image = image.numpy()
        image = image / 255.
        image_augment_method_index = random.randint(1, 3)
        if image_augment_method_index == 1:
            image = cv.blur(src=image, ksize=(3, 3))
        elif image_augment_method_index == 2:
            image = cv.flip(src=image, flipCode=0)
        else:
            image = cv.rotate(src=image, rotateCode=cv.ROTATE_90_CLOCKWISE)
        image = tf.convert_to_tensor(image, dtype=tf.float32)
        label = tf.convert_to_tensor(label, dtype=tf.int32)

        return image, label
