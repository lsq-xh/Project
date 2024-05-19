"""
说明：
    搭建Inception v1 v2 v3网络,由于V2网络只是V1版本添加BN层和辅组分类器，因此并未搭建V2版本模型
"""
from tensorflow.keras import Input
from tensorflow.keras import Model
from tensorflow.keras import layers


class Inception:
    def __init__(self, number_of_classes, image_shape):
        """
        对网络进行初始化
        :param number_of_classes: 数据集识别对象种类
        """
        self.kind_of_classes = number_of_classes
        self.image_shape = image_shape

    """ 用于inception v1"""

    @staticmethod
    def inception_module_v1(filter_1, filter_2, filter_3, filter_4, filter_5, filter_6, in_tensor):
        """
        inception v1 中的 inception 模块
        :param filter_1: feature map 数量，具体数值参见论文
        :param filter_2: feature map 数量，具体数值参见论文
        :param filter_3: feature map 数量，具体数值参见论文
        :param filter_4: feature map 数量，具体数值参见论文
        :param filter_5: feature map 数量，具体数值参见论文
        :param filter_6: feature map 数量，具体数值参见论文
        :param in_tensor: 输入张量
        :return: 输出张量
        """
        x = in_tensor
        branch_1 = layers.Conv2D(filters=filter_1, kernel_size=(1, 1), strides=(1, 1), padding="same", activation="relu")(x)
        branch_2 = layers.Conv2D(filters=filter_2, kernel_size=(1, 1), strides=(1, 1), padding="same", activation="relu")(x)
        branch_2 = layers.Conv2D(filters=filter_3, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(branch_2)
        branch_3 = layers.Conv2D(filters=filter_4, kernel_size=(1, 1), strides=(1, 1), padding="same", activation="relu")(x)
        branch_3 = layers.Conv2D(filters=filter_5, kernel_size=(5, 5), strides=(1, 1), padding="same", activation="relu")(branch_3)
        branch_4 = layers.MaxPool2D(pool_size=(3, 3), strides=(1, 1), padding="same")(x)
        branch_4 = layers.Conv2D(filters=filter_6, kernel_size=(1, 1), strides=(1, 1), padding="same", activation="relu")(branch_4)

        out_tensor = layers.Concatenate(axis=-1)([branch_1, branch_2, branch_3, branch_4])

        return out_tensor

    """ 用于inception v1"""

    """ 用于inception v3"""

    @staticmethod
    def inception_module_v3_a(filters, in_tensor):
        """
        inception v3 中的 inception 模块
        :param filters: feature map 数量，具体数值参见论文
        :param in_tensor: 输入张量
        :return: 输出张量
        """
        branch_1 = layers.Conv2D(filters=filters[0], kernel_size=(1, 1), strides=(1, 1), padding="same", activation="relu")(in_tensor)
        branch_1 = layers.BatchNormalization()(branch_1)
        branch_1 = layers.Conv2D(filters=filters[1], kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(branch_1)
        branch_1 = layers.BatchNormalization(branch_1)
        branch_1 = layers.Conv2D(filters=filters[2], kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(branch_1)
        branch_1 = layers.BatchNormalization()(branch_1)

        branch_2 = layers.Conv2D(filters=filters[3], kernel_size=(1, 1), strides=(1, 1), padding="same", activation="relu")(in_tensor)
        branch_2 = layers.BatchNormalization()(branch_2)
        branch_2 = layers.Conv2D(filters=filters[4], kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(branch_2)
        branch_2 = layers.BatchNormalization()(branch_2)

        branch_3 = layers.MaxPool2D(pool_size=(3, 3), strides=(1, 1), padding="same")(in_tensor)
        branch_3 = layers.Conv2D(filters=filters[5], kernel_size=(1, 1), strides=(1, 1), padding="same", activation="relu")(branch_3)
        branch_3 = layers.BatchNormalization()(branch_3)

        branch_4 = layers.Conv2D(filters=filters[6], kernel_size=(1, 1), strides=(1, 1), padding="same", activation="relu")(in_tensor)
        branch_4 = layers.BatchNormalization()(branch_4)

        out_tensor = layers.Concatenate(axis=-1)([branch_1, branch_2, branch_3, branch_4])
        return out_tensor

    @staticmethod
    def inception_module_v3_b(filters, in_tensor):
        """
        inception v3 中的 inception 模块
        :param filters: feature map 数量，具体数值参见论文
        :param in_tensor: 输入张量
        :return: 输出张量
        """
        branch_1 = layers.Conv2D(filters=filters[0], kernel_size=(1, 1), strides=(1, 1), padding="same", activation="relu")(in_tensor)
        branch_1 = layers.BatchNormalization()(branch_1)
        branch_1 = layers.Conv2D(filters=filters[1], kernel_size=(1, 7), strides=(1, 1), padding="same", activation="relu")(branch_1)
        branch_1 = layers.BatchNormalization()(branch_1)
        branch_1 = layers.Conv2D(filters=filters[2], kernel_size=(7, 1), strides=(1, 1), padding="same", activation="relu")(branch_1)
        branch_1 = layers.BatchNormalization()(branch_1)
        branch_1 = layers.Conv2D(filters=filters[3], kernel_size=(1, 7), strides=(1, 1), padding="same", activation="relu")(branch_1)
        branch_1 = layers.BatchNormalization()(branch_1)
        branch_1 = layers.Conv2D(filters=filters[4], kernel_size=(7, 1), strides=(1, 1), padding="same", activation="relu")(branch_1)
        branch_1 = layers.BatchNormalization()(branch_1)

        branch_2 = layers.Conv2D(filters=filters[5], kernel_size=(1, 1), strides=(1, 1), padding="same", activation="relu")(in_tensor)
        branch_2 = layers.BatchNormalization()(branch_2)
        branch_2 = layers.Conv2D(filters=filters[6], kernel_size=(1, 7), strides=(1, 1), padding="same", activation="relu")(branch_2)
        branch_2 = layers.BatchNormalization()(branch_2)
        branch_2 = layers.Conv2D(filters=filters[7], kernel_size=(7, 1), strides=(1, 1), padding="same", activation="relu")(branch_2)
        branch_2 = layers.BatchNormalization()(branch_2)

        branch_3 = layers.MaxPool2D(pool_size=(3, 3), strides=(1, 1), padding="same")(in_tensor)
        branch_3 = layers.Conv2D(filters=filters[8], kernel_size=(1, 1), strides=(1, 1), padding="same", activation="relu")(branch_3)
        branch_3 = layers.BatchNormalization()(branch_3)

        branch_4 = layers.Conv2D(filters=filters[9], kernel_size=(1, 1), strides=(1, 1), padding="same", activation="relu")(in_tensor)
        branch_4 = layers.BatchNormalization()(branch_4)

        out_tensor = layers.Concatenate(axis=-1)([branch_1, branch_2, branch_3, branch_4])
        return out_tensor

    @staticmethod
    def inception_module_v3_c(filters, in_tensor):
        """
        inception v3 中的 inception 模块
        :param filters: feature map 数量，具体数值参见论文
        :param in_tensor: 输入张量
        :return: 输出张量
        """
        branch_1 = layers.Conv2D(filters=filters[0], kernel_size=(1, 1), strides=(1, 1), padding="same", activation="relu")(in_tensor)
        branch_1 = layers.BatchNormalization()(branch_1)
        branch_1 = layers.Conv2D(filters=filters[1], kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(branch_1)
        branch_1 = layers.BatchNormalization()(branch_1)
        branch_1_1 = layers.Conv2D(filters=filters[2], kernel_size=(1, 3), strides=(1, 1), padding="same", activation="relu")(branch_1)
        branch_1_1 = layers.BatchNormalization()(branch_1_1)
        branch_1_2 = layers.Conv2D(filters=filters[3], kernel_size=(3, 1), strides=(1, 1), padding="same", activation="relu")(branch_1)
        branch_1_2 = layers.BatchNormalization()(branch_1_2)
        branch_1 = layers.Concatenate(axis=-1)([branch_1_1, branch_1_2])

        branch_2 = layers.Conv2D(filters=filters[4], kernel_size=(1, 1), strides=(1, 1), padding="same", activation="relu")(in_tensor)
        branch_2 = layers.BatchNormalization()(branch_2)
        branch_2_1 = layers.Conv2D(filters=filters[5], kernel_size=(1, 3), strides=(1, 1), padding="same", activation="relu")(branch_2)
        branch_2_1 = layers.BatchNormalization()(branch_2_1)
        branch_2_2 = layers.Conv2D(filters=filters[6], kernel_size=(3, 1), strides=(1, 1), padding="same", activation="relu")(branch_2)
        branch_2_2 = layers.BatchNormalization()(branch_2_2)
        branch_2 = layers.Concatenate(axis=-1)([branch_2_1, branch_2_2])

        branch_3 = layers.MaxPool2D(pool_size=(3, 3), strides=(1, 1), padding="same")(in_tensor)
        branch_3 = layers.Conv2D(filters=filters[7], kernel_size=(1, 1), strides=(1, 1), padding="same", activation="relu")(branch_3)
        branch_3 = layers.BatchNormalization()(branch_3)

        branch_4 = layers.Conv2D(filters=filters[8], kernel_size=(1, 1), strides=(1, 1), padding="same", activation="relu")(in_tensor)
        branch_4 = layers.BatchNormalization()(branch_4)
        out_tensor = layers.Concatenate(axis=-1)([branch_1, branch_2, branch_3, branch_4])

        return out_tensor

    """ 用于inception v3"""

    """ 用于inception v4"""

    @staticmethod
    def stem(in_tensor):
        """
        inception v4 中的 stem 模块
        :param in_tensor: 输入张量
        :return: 输出张量
        """
        x = layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), padding="valid", activation="relu")(in_tensor)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding="valid", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="valid", activation="relu")(x)
        x = layers.BatchNormalization()(x)

        x_branch_1_1 = layers.Conv2D(filters=96, kernel_size=(3, 3), strides=(2, 2), padding="valid", activation="relu")(x)
        x_branch_1_1 = layers.BatchNormalization()(x_branch_1_1)

        x_branch_1_2 = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="valid")(x)
        x = layers.Concatenate(axis=-1)([x_branch_1_1, x_branch_1_2])

        x_branch_2_1 = layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding="same", activation="relu")(x)
        x_branch_2_1 = layers.BatchNormalization()(x_branch_2_1)
        x_branch_2_1 = layers.Conv2D(filters=96, kernel_size=(3, 3), strides=(1, 1), padding="valid", activation="relu")(x_branch_2_1)
        x_branch_2_1 = layers.BatchNormalization()(x_branch_2_1)
        x_branch_2_2 = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(x)
        x_branch_2_2 = layers.BatchNormalization()(x_branch_2_2)
        x_branch_2_2 = layers.Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding="same", activation="relu")(x_branch_2_2)
        x_branch_2_2 = layers.BatchNormalization()(x_branch_2_2)
        x_branch_2_2 = layers.Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding="same", activation="relu")(x_branch_2_2)
        x_branch_2_2 = layers.BatchNormalization()(x_branch_2_2)
        x_branch_2_2 = layers.Conv2D(filters=96, kernel_size=(3, 3), strides=(1, 1), padding="valid", activation="relu")(x_branch_2_2)
        x_branch_2_2 = layers.BatchNormalization()(x_branch_2_2)
        x = layers.Concatenate(axis=-1)([x_branch_2_1, x_branch_2_2])

        x_branch_3_1 = layers.Conv2D(filters=192, kernel_size=(3, 3), strides=(2, 2), padding="same", activation="relu")(x)
        x_branch_3_1 = layers.BatchNormalization()(x_branch_3_1)

        x_branch_3_2 = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="same")(x)
        x = layers.Concatenate(axis=-1)([x_branch_3_1, x_branch_3_2])
        out_tensor = x

        return out_tensor

    @staticmethod
    def inception_a(in_tensor):
        """
        inception v4 中的 inception 模块
        :param in_tensor: 输入张量
        :return: 输出张量
        """
        x = in_tensor
        x_branch_1 = layers.AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding="same")(x)
        x_branch_1 = layers.Conv2D(filters=96, kernel_size=(1, 1), strides=(1, 1), padding="same", activation="relu")(x_branch_1)
        x_branch_1 = layers.BatchNormalization()(x_branch_1)

        x_branch_2 = layers.Conv2D(filters=96, kernel_size=(1, 1), strides=(1, 1), padding="same", activation="relu")(x)
        x_branch_2 = layers.BatchNormalization()(x_branch_2)

        x_branch_3 = layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding="same", activation="relu")(x)
        x_branch_3 = layers.BatchNormalization()(x_branch_3)
        x_branch_3 = layers.Conv2D(filters=96, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(x_branch_3)
        x_branch_3 = layers.BatchNormalization()(x_branch_3)

        x_branch_4 = layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding="same", activation="relu")(x)
        x_branch_4 = layers.BatchNormalization()(x_branch_4)
        x_branch_4 = layers.Conv2D(filters=96, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(x_branch_4)
        x_branch_4 = layers.BatchNormalization()(x_branch_4)
        x_branch_4 = layers.Conv2D(filters=96, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(x_branch_4)
        x_branch_4 = layers.BatchNormalization()(x_branch_4)

        out_tensor = layers.Concatenate(axis=-1)([x_branch_1, x_branch_2, x_branch_3, x_branch_4])
        return out_tensor

    @staticmethod
    def reduction_a(k, i, m, n, in_tensor):
        """
        inception v4 中的 reduction 模块，具体原理参见论文
        :param k: feature map 数量，具体数值参见论文
        :param i: feature map 数量，具体数值参见论文
        :param m: feature map 数量，具体数值参见论文
        :param n: feature map 数量，具体数值参见论文
        :param in_tensor: 输入张量
        :return: 输出张量
        """
        x = in_tensor
        branch_1 = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="same")(x)

        branch_2 = layers.Conv2D(filters=n, kernel_size=(3, 3), strides=(2, 2), padding="same", activation="relu")(x)
        branch_2 = layers.BatchNormalization()(branch_2)

        branch_3 = layers.Conv2D(filters=k, kernel_size=(1, 1), strides=(1, 1), padding="same", activation="relu")(x)
        branch_3 = layers.BatchNormalization()(branch_3)
        branch_3 = layers.Conv2D(filters=i, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(branch_3)
        branch_3 = layers.BatchNormalization()(branch_3)
        branch_3 = layers.Conv2D(filters=m, kernel_size=(3, 3), strides=(2, 2), padding="same", activation="relu")(branch_3)
        branch_3 = layers.BatchNormalization()(branch_3)

        out_tensor = layers.Concatenate(axis=-1)([branch_1, branch_2, branch_3])
        return out_tensor

    @staticmethod
    def inception_b(in_tensor):
        """
        inception v4 中的 inception 模块
        :param in_tensor: 输入张量
        :return: 输出张量
        """
        x = in_tensor
        x_branch_1 = layers.AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding="same")(x)
        x_branch_1 = layers.Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding="same", activation="relu")(x_branch_1)
        x_branch_1 = layers.BatchNormalization()(x_branch_1)

        x_branch_2 = layers.Conv2D(filters=384, kernel_size=(1, 1), strides=(1, 1), padding="same", activation="relu")(x)
        x_branch_2 = layers.BatchNormalization()(x_branch_2)

        x_branch_3 = layers.Conv2D(filters=192, kernel_size=(1, 1), strides=(1, 1), padding="same", activation="relu")(x)
        x_branch_3 = layers.BatchNormalization()(x_branch_3)
        x_branch_3 = layers.Conv2D(filters=224, kernel_size=(1, 7), strides=(1, 1), padding="same", activation="relu")(x_branch_3)
        x_branch_3 = layers.BatchNormalization()(x_branch_3)
        x_branch_3 = layers.Conv2D(filters=256, kernel_size=(1, 7,), strides=(1, 1), padding="same", activation="relu")(x_branch_3)
        x_branch_3 = layers.BatchNormalization()(x_branch_3)

        x_branch_4 = layers.Conv2D(filters=192, kernel_size=(1, 1), strides=(1, 1), padding="same", activation="relu")(x)
        x_branch_4 = layers.BatchNormalization()(x_branch_4)
        x_branch_4 = layers.Conv2D(filters=192, kernel_size=(1, 7), strides=(1, 1), padding="same", activation="relu")(x_branch_4)
        x_branch_4 = layers.BatchNormalization()(x_branch_4)
        x_branch_4 = layers.Conv2D(filters=224, kernel_size=(7, 1), strides=(1, 1), padding="same", activation="relu")(x_branch_4)
        x_branch_4 = layers.BatchNormalization()(x_branch_4)
        x_branch_4 = layers.Conv2D(filters=224, kernel_size=(1, 7), strides=(1, 1), padding="same", activation="relu")(x_branch_4)
        x_branch_4 = layers.BatchNormalization()(x_branch_4)
        x_branch_4 = layers.Conv2D(filters=256, kernel_size=(7, 1), strides=(1, 1), padding="same", activation="relu")(x_branch_4)
        x_branch_4 = layers.BatchNormalization()(x_branch_4)

        out_tensor = layers.Concatenate(axis=-1)([x_branch_1, x_branch_2, x_branch_3, x_branch_4])

        return out_tensor

    @staticmethod
    def reduction_b(in_tensor):
        """
          inception v4 中的 reduction 模块，具体原理参见论文
        :param in_tensor: 输入张量
        :return: 输出张量
        """
        x = in_tensor
        branch_1 = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="same")(x)

        branch_2 = layers.Conv2D(filters=192, kernel_size=(1, 1), strides=(1, 1), padding="same", activation="relu")(x)
        branch_2 = layers.BatchNormalization()(branch_2)
        branch_2 = layers.Conv2D(filters=192, kernel_size=(3, 3), strides=(2, 2), padding="same", activation="relu")(branch_2)
        branch_2 = layers.BatchNormalization()(branch_2)

        branch_3 = layers.Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding="same", activation="relu")(x)
        branch_3 = layers.BatchNormalization()(branch_3)
        branch_3 = layers.Conv2D(filters=256, kernel_size=(1, 7), strides=(1, 1), padding="same", activation="relu")(branch_3)
        branch_3 = layers.BatchNormalization()(branch_3)
        branch_3 = layers.Conv2D(filters=320, kernel_size=(7, 1), strides=(1, 1), padding="same", activation="relu")(branch_3)
        branch_3 = layers.BatchNormalization()(branch_3)
        branch_3 = layers.Conv2D(filters=320, kernel_size=(3, 3), strides=(2, 2), padding="same", activation="relu")(branch_3)
        branch_3 = layers.BatchNormalization()(branch_3)

        out_tensor = layers.Concatenate(axis=-1)([branch_1, branch_2, branch_3])
        return out_tensor

    @staticmethod
    def inception_c(in_tensor):
        """
        inception v4 中的 inception 模块
        :param in_tensor: 输入张量
        :return: 输出张量
        """
        x = in_tensor
        x_branch_1 = layers.AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding="same")(x)
        x_branch_1 = layers.Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding="same", activation="relu")(x_branch_1)
        x_branch_1 = layers.BatchNormalization()(x_branch_1)

        x_branch_2 = layers.Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding="same", activation="relu")(x)
        x_branch_2 = layers.BatchNormalization()(x_branch_2)

        x_branch_3 = layers.Conv2D(filters=384, kernel_size=(1, 1), strides=(1, 1), padding="same", activation="relu")(x)
        x_branch_3 = layers.BatchNormalization()(x_branch_3)
        x_branch_3_1 = layers.Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding="same", activation="relu")(x_branch_3)
        x_branch_3_1 = layers.BatchNormalization()(x_branch_3_1)
        x_branch_3_2 = layers.Conv2D(filters=256, kernel_size=(3, 1), strides=(1, 1), padding="same", activation="relu")(x_branch_3)
        x_branch_3_2 = layers.BatchNormalization()(x_branch_3_2)
        x_branch_3 = layers.Concatenate(axis=-1)([x_branch_3_1, x_branch_3_2])

        x_branch_4 = layers.Conv2D(filters=384, kernel_size=(1, 1), strides=(1, 1), padding="same", activation="relu")(x)
        x_branch_4 = layers.BatchNormalization()(x_branch_4)
        x_branch_4 = layers.Conv2D(filters=448, kernel_size=(1, 3), strides=(1, 1), padding="same", activation="relu")(x_branch_4)
        x_branch_4 = layers.BatchNormalization()(x_branch_4)
        x_branch_4 = layers.Conv2D(filters=512, kernel_size=(3, 1), strides=(1, 1), padding="same", activation="relu")(x_branch_4)
        x_branch_4 = layers.BatchNormalization()(x_branch_4)
        x_branch_4_1 = layers.Conv2D(filters=256, kernel_size=(3, 1), strides=(1, 1), padding="same", activation="relu")(x_branch_4)
        x_branch_4_1 = layers.BatchNormalization()(x_branch_4_1)
        x_branch_4_2 = layers.Conv2D(filters=256, kernel_size=(1, 3), strides=(1, 1), padding="same", activation="relu")(x_branch_4)
        x_branch_4_2 = layers.BatchNormalization()(x_branch_4_2)
        x_branch_4 = layers.Concatenate(axis=-1)([x_branch_4_1, x_branch_4_2])

        out_tensor = layers.Concatenate(axis=-1)([x_branch_1, x_branch_2, x_branch_3, x_branch_4])
        return out_tensor

    """ 用于inception v4"""

    def inception_net(self, version):
        """
        inception 网络模型
        :param version: 选择所需要的 inception 版本
        :return: 网络模型
        """
        inputs = Input(shape=self.image_shape)
        if version == 1:
            x = layers.Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), padding="same", activation="relu")(inputs)
            x = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="same")(x)
            x = layers.Conv2D(filters=192, kernel_size=(3, 3), strides=(2, 2), padding="same", activation="relu")(x)
            x = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="same")(x)
            x = self.inception_module_v1(filter_1=64, filter_2=96, filter_3=128, filter_4=16, filter_5=32, filter_6=32, in_tensor=x)
            x = self.inception_module_v1(filter_1=128, filter_2=128, filter_3=192, filter_4=32, filter_5=96, filter_6=64, in_tensor=x)
            x = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="same")(x)
            x = self.inception_module_v1(filter_1=192, filter_2=96, filter_3=208, filter_4=16, filter_5=48, filter_6=64, in_tensor=x)
            x = self.inception_module_v1(filter_1=160, filter_2=112, filter_3=224, filter_4=24, filter_5=64, filter_6=64, in_tensor=x)
            x = self.inception_module_v1(filter_1=128, filter_2=128, filter_3=256, filter_4=24, filter_5=64, filter_6=64, in_tensor=x)
            x = self.inception_module_v1(filter_1=112, filter_2=144, filter_3=288, filter_4=32, filter_5=64, filter_6=64, in_tensor=x)
            x = self.inception_module_v1(filter_1=256, filter_2=160, filter_3=320, filter_4=32, filter_5=128, filter_6=128, in_tensor=x)
            x = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="same")(x)
            x = self.inception_module_v1(filter_1=256, filter_2=160, filter_3=320, filter_4=32, filter_5=128, filter_6=128, in_tensor=x)
            x = self.inception_module_v1(filter_1=384, filter_2=192, filter_3=384, filter_4=48, filter_5=128, filter_6=128, in_tensor=x)
            x = layers.AveragePooling2D(pool_size=(7, 7), strides=(1, 1), padding="same")(x)
            x = layers.Dropout(rate=0.4)(x)
            x = layers.Flatten()(x)
            outputs = layers.Dense(units=self.kind_of_classes, activation="softmax")(x)
            model = Model(inputs=inputs, outputs=outputs)
            return model
        elif version == 2:
            pass
        elif version == 3:
            x = layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), padding="same", activation="relu")(inputs)
            x = layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(x)
            x = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(x)
            x = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="same")(x)
            x = layers.Conv2D(filters=80, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(x)
            x = layers.Conv2D(filters=192, kernel_size=(3, 3), strides=(2, 2), padding="same", activation="relu")(x)
            x = layers.Conv2D(filters=288, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(x)
            x = self.inception_module_v3_a(filters=[64, 96, 96, 48, 64, 32, 64], in_tensor=x)
            x = self.inception_module_v3_a(filters=[64, 96, 96, 48, 64, 32, 64], in_tensor=x)
            x = self.inception_module_v3_a(filters=[64, 96, 96, 48, 64, 32, 64], in_tensor=x)
            x = self.inception_module_v3_b(filters=[128, 128, 128, 128, 192, 128, 128, 192, 192, 192], in_tensor=x)
            x = self.inception_module_v3_b(filters=[128, 128, 128, 128, 192, 128, 128, 192, 192, 192], in_tensor=x)
            x = self.inception_module_v3_b(filters=[128, 128, 128, 128, 192, 128, 128, 192, 192, 192], in_tensor=x)
            x = self.inception_module_v3_b(filters=[128, 128, 128, 128, 192, 128, 128, 192, 192, 192], in_tensor=x)
            x = self.inception_module_v3_b(filters=[128, 128, 128, 128, 192, 128, 128, 192, 192, 192], in_tensor=x)
            x = self.inception_module_v3_c(filters=[448, 384, 384, 384, 384, 384, 384, 192, 320], in_tensor=x)
            x = self.inception_module_v3_c(filters=[448, 384, 384, 384, 384, 384, 384, 192, 320], in_tensor=x)
            x = layers.MaxPool2D(pool_size=(8, 8), strides=(1, 1), padding="same")(x)
            x = layers.Dropout(rate=0.4)(x)
            x = layers.Flatten()(x)
            outputs = layers.Dense(units=self.kind_of_classes, activation="softmax")(x)
            model = Model(inputs=inputs, outputs=outputs)
            return model
        elif version == 4:
            x = self.stem(in_tensor=inputs)
            for _ in range(4):
                x = self.inception_a(in_tensor=x)
            x = self.reduction_a(k=192, i=224, m=256, n=384, in_tensor=x)
            for _ in range(7):
                x = self.inception_b(in_tensor=x)
            x = self.reduction_b(in_tensor=x)
            for _ in range(3):
                x = self.inception_c(in_tensor=x)
            x = layers.AveragePooling2D(pool_size=(8, 8), strides=(1, 1), padding="same")(x)
            x = layers.Dropout(rate=0.8)(x)
            x = layers.Flatten()(x)
            outputs = layers.Dense(units=self.kind_of_classes, activation="softmax")(x)
            model = Model(inputs=inputs, outputs=outputs)
            return model

        else:
            print("please enter the right version number, it can be 1、2、3、4。")
