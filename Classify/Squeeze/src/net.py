"""
说明：
    搭建VGG网络
"""
from tensorflow.keras import Input
from tensorflow.keras import Model
from tensorflow.keras import layers


class Squeeze:
    def __init__(self, number_of_classes, image_shape):
        """
        对网络进行初始化
        :param number_of_classes: 数据集识别对象种类
        """
        self.kind_of_classes = number_of_classes
        self.image_shape = image_shape

    @staticmethod
    def fire_module(s1, e_left, e_right, in_tensor):
        """
        论文中所提到的fire 模块
        :param s1: feature map 数量，具体数值及原理参见论文
        :param e_left: feature map 数量，具体数值及原理参见论文
        :param e_right: feature map 数量，具体数值及原理参见论文
        :param in_tensor: 输入张量
        :return: 输出张量
        """
        squeeze = layers.Conv2D(filters=s1, kernel_size=(1, 1), strides=(1, 1), padding="same", activation="relu")(in_tensor)
        expand_left = layers.Conv2D(filters=e_left, kernel_size=(1, 1), strides=(1, 1), padding="same", activation="relu")(squeeze)
        expand_right = layers.Conv2D(filters=e_right, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(squeeze)
        out_tensor = layers.Concatenate(axis=-1)([expand_left, expand_right])

        return out_tensor

    def squeeze_net(self):
        """
        squeeze net 网络模型搭建
        :return: 网络模型
        """
        inputs = Input(shape=self.image_shape)
        x = layers.Conv2D(filters=96, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(inputs)
        x = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same")(x)
        fire2 = self.fire_module(s1=16, e_left=64, e_right=64, in_tensor=x)
        fire3 = self.fire_module(s1=16, e_left=64, e_right=64, in_tensor=fire2)
        fire4 = self.fire_module(s1=32, e_left=128, e_right=128, in_tensor=fire3)
        x = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same")(fire4)
        fire5 = self.fire_module(s1=32, e_left=128, e_right=128, in_tensor=x)
        fire6 = self.fire_module(s1=48, e_left=192, e_right=192, in_tensor=fire5)
        fire7 = self.fire_module(s1=48, e_left=192, e_right=192, in_tensor=fire6)
        fire8 = self.fire_module(s1=64, e_left=256, e_right=256, in_tensor=fire7)
        x = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same")(fire8)
        fire9 = self.fire_module(s1=64, e_left=256, e_right=256, in_tensor=x)
        x = layers.Conv2D(filters=1000, kernel_size=(1, 1), strides=(1, 1), padding="same", activation="relu")(fire9)
        x = layers.AveragePooling2D(pool_size=(13, 13), strides=(1, 1), padding="same")(x)
        x = layers.Flatten()(x)
        outputs = layers.Dense(units=self.kind_of_classes, activation="softmax")(x)

        model = Model(inputs=inputs, outputs=outputs)

        return model
