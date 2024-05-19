"""
说明：
    搭建VGG网络
"""

import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras import Model
from tensorflow.keras import layers


class Shuffle:
    def __init__(self, number_of_classes, image_shape):
        """
        对网络进行初始化
        :param number_of_classes: 数据集识别对象种类
        """
        self.kind_of_classes = number_of_classes
        self.image_shape = image_shape

    """ 用于shuffle net v1 的网络模块"""

    @staticmethod
    def channel_shuffle(in_tensor, group):
        """
        shuffle net 对通道进行 shuffle操作
        :param in_tensor: 输入张量
        :param group: feature map 分组数量，具体参见论文, 代码中默认选取 group 等于 8
        :return: 输出张量
        """
        x = in_tensor
        print(x.shape)
        print("-----------")
        n, h, w, c = x.shape
        x = tf.reshape(tensor=x, shape=[-1, h, w, group, c // group])
        x = tf.transpose(a=x, perm=[0, 1, 2, 4, 3])
        out_tensor = tf.reshape(tensor=x, shape=[-1, h, w, c])
        print(out_tensor.shape)

        return out_tensor

    @staticmethod
    def group_conv(in_tensor, group, filters):
        """
        shuffle net 组卷积操作
        :param in_tensor: 输入张量
        :param group:  feature map 分组数量，具体参见论文，代码中默认选取 group 等于 8
        :param filters: feature map 数量
        :return: 输出张量
        """
        x = in_tensor
        x_group_layer = tf.split(value=x, num_or_size_splits=group, axis=-1)
        x_group_layer_convolution = []
        for layer in x_group_layer:
            x = layers.Conv2D(filters=filters // 4, kernel_size=(1, 1), strides=(1, 1), padding="same")(layer)
            x_group_layer_convolution.append(x)

        output = layers.Concatenate(axis=-1)(x_group_layer_convolution)
        return output

    def shuffle_unit_stride_2(self, in_tensor, group, filters):
        """
        stride 等于 2 时使用的 shuffle unit
        :param in_tensor: 输入张量
        :param group:  feature map 分组数量，具体参见论文，代码中默认选取 group 等于 8
        :param filters:  feature map 数量
        :return: 输出张量
        """
        x = in_tensor
        x_branch_1 = layers.AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(x)
        x_branch_2 = self.group_conv(in_tensor=x, group=group, filters=filters)
        x_branch_2 = layers.BatchNormalization()(x_branch_2)
        x_branch_2 = layers.Activation("relu")(x_branch_2)
        x_branch_2 = self.channel_shuffle(in_tensor=x_branch_2, group=group)
        x_branch_2 = layers.DepthwiseConv2D(kernel_size=(3, 3), strides=(2, 2), padding="same")(x_branch_2)
        x_branch_2 = layers.BatchNormalization()(x_branch_2)
        x_branch_2 = self.group_conv(in_tensor=x_branch_2, group=group, filters=filters)
        x_branch_2 = layers.BatchNormalization()(x_branch_2)
        out_tensor = layers.Concatenate(axis=-1)([x_branch_1, x_branch_2])

        return out_tensor

    def shuffle_unit_stride_1(self, in_tensor, group, filters):
        """
        stride 等于 1 时使用的 shuffle unit
        :param in_tensor: 输入张量
        :param group: feature map 分组数量，具体参见论文，代码中默认选取 group 等于 8
        :param filters: feature map 数量
        :return: 输出张量
        """
        x = in_tensor
        x = self.group_conv(in_tensor=x, group=group, filters=filters)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = self.channel_shuffle(in_tensor=x, group=group)
        x = layers.DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = self.group_conv(in_tensor=x, group=group, filters=filters)
        x = layers.BatchNormalization()(x)
        out_tensor = layers.Add()([in_tensor, x])
        return out_tensor

    def stage(self, shuffle_unit_repeat_time, in_tensor, group, filters):
        """
        由 shuffle unit 构建的 stage 模块
        :param shuffle_unit_repeat_time: stride = 1 的 shuffle unit 重复的次数
        :param in_tensor: 输入张量
        :param group:feature map 分组数量，具体参见论文
        :param filters:feature map 数量
        :return: 输出张量
        """
        x = in_tensor
        x = self.shuffle_unit_stride_2(in_tensor=x, group=group, filters=filters)
        for _ in range(shuffle_unit_repeat_time):
            x = self.shuffle_unit_stride_1(in_tensor=x, group=group, filters=filters)
            print("++++")

        out_tensor = x
        return out_tensor

    """ 用于shuffle net v1 的网络模块"""

    def shuffle_net(self):
        """
        shuffle net 模型搭建
        :return: 网络模型
        """
        inputs = Input(shape=self.image_shape)
        x = layers.Conv2D(filters=112, kernel_size=(3, 3), strides=(2, 2), padding="same", activation="relu")(inputs)
        x = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="same")(x)
        x = self.stage(shuffle_unit_repeat_time=3, in_tensor=x, group=8, filters=384)
        x = self.stage(shuffle_unit_repeat_time=7, in_tensor=x, group=8, filters=768)
        x = self.stage(shuffle_unit_repeat_time=3, in_tensor=x, group=8, filters=1536)
        x = layers.GlobalAvgPool2D()(x)
        flatten = layers.Flatten()(x)
        outputs = layers.Dense(units=self.kind_of_classes, activation="softmax")(flatten)
        model = Model(inputs=inputs, outputs=outputs)

        return model
