"""
说明：
    搭建VGG网络
"""
from tensorflow.keras import Input
from tensorflow.keras import Model
from tensorflow.keras import layers


class Dense:
    def __init__(self, number_of_classes, image_shape):
        """
        对网络进行初始化
        :param number_of_classes: 数据集识别对象种类
        """
        self.kind_of_classes = number_of_classes
        self.image_shape = image_shape

    @staticmethod
    def transition_block(filters, in_tensor):
        """
        构建Dense net 的transition模块
        :param filters: feature map 数量
        :param in_tensor: 输入张量
        :return: 输出张量
        """
        x = layers.BatchNormalization()(in_tensor)
        x = layers.Conv2D(filters=filters, kernel_size=(1, 1), strides=(1, 1), padding="same", activation="relu")(x)
        out_tensor = layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(x)

        return out_tensor

    @staticmethod
    def conv_block(filters, in_tensor):
        """
        构建基本的卷积块
        :param filters: feature map 数量
        :param in_tensor:  输入张量
        :return: 输出张量
        """
        x = layers.BatchNormalization()(in_tensor)
        x = layers.Conv2D(filters=filters * 4, kernel_size=(1, 1), strides=(1, 1), padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(x)
        x = layers.Concatenate(axis=-1)([x, in_tensor])
        out_tensor = x

        return out_tensor

    def dense_block(self, filters, layer_number, in_tensor):
        """
        构建网络的dense 模块
        :param filters: feature map 数量
        :param layer_number: 基本卷积块循环次数
        :param in_tensor: 输入张量
        :return: 输出张量
        """
        x = in_tensor
        for _ in range(layer_number):
            x = self.conv_block(filters=filters, in_tensor=x)
        outputs = x

        return outputs

    def dense_net_bc(self, layers_number, k, theta):
        """
        搭建dense net 网络模型
        :param layers_number: 基本卷积块循环次数
        :param k: 论文中的超参数k，主要决定每一层输出的feature map的数量
        :param theta: 论文中的超参数theta, 主要决定transition layer的feature map的数量
        :return: densenet网络模型
        """
        dense_layer_block = []
        if layers_number == 121:
            dense_layer_block = [6, 12, 24, 16]
        elif layers_number == 169:
            dense_layer_block = [6, 12, 32, 32]
        elif layers_number == 201:
            dense_layer_block = [6, 12, 48, 32]
        elif layers_number == 264:
            dense_layer_block = [6, 12, 64, 48]
        else:
            print("please enter the right number,it can be 121、169、201、264")
        inputs = Input(shape=self.image_shape)
        x = layers.Conv2D(filters=2 * k, kernel_size=(7, 7), strides=(2, 2), padding="same", activation="relu")(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="same")(x)
        x = self.dense_block(filters=k, layer_number=dense_layer_block[0], in_tensor=x)
        x = self.transition_block(filters=int(x.shape[-1] * theta), in_tensor=x)
        x = self.dense_block(filters=k, layer_number=dense_layer_block[1], in_tensor=x)
        x = self.transition_block(filters=int(x.shape[-1] * theta), in_tensor=x)
        x = self.dense_block(filters=k, layer_number=dense_layer_block[2], in_tensor=x)
        x = self.transition_block(filters=int(x.shape[-1] * theta), in_tensor=x)
        x = self.dense_block(filters=k, layer_number=dense_layer_block[3], in_tensor=x)
        x = layers.GlobalAvgPool2D()(x)
        x = layers.Flatten()(x)
        outputs = layers.Dense(units=self.kind_of_classes, activation="softmax")(x)

        model = Model(inputs=inputs, outputs=outputs)

        return model
