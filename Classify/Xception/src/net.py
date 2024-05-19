"""
说明：
    搭建VGG网络
"""
from tensorflow.keras import Input
from tensorflow.keras import Model
from tensorflow.keras import layers


class XCEPTION:
    def __init__(self, number_of_classes, image_shape):
        """
        对网络进行初始化
        :param number_of_classes: 数据集识别对象种类
        """
        self.kind_of_classes = number_of_classes
        self.image_shape = image_shape

    @staticmethod
    def entry_flow(inputs):
        """
        xception 网络的 entry flow 模块
        :param inputs: 输入张量
        :return: 输出张量
        """
        x = layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), padding="same", activation="relu")(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x1_1 = layers.SeparableConv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(x)
        x1_1 = layers.BatchNormalization()(x1_1)
        x1_1 = layers.SeparableConv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(x1_1)
        x1_1 = layers.BatchNormalization()(x1_1)
        x1_1 = layers.MaxPool2D(pool_size=(2, 2), padding="same", strides=(2, 2))(x1_1)
        x1_2 = layers.Conv2D(filters=128, kernel_size=(1, 1), strides=(2, 2), padding="same", activation="relu")(x)
        x1_2 = layers.BatchNormalization()(x1_2)
        x1 = layers.Add()([x1_2, x1_1])

        x2_1 = layers.SeparableConv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(x1)
        x2_1 = layers.BatchNormalization()(x2_1)
        x2_1 = layers.SeparableConv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(x2_1)
        x2_1 = layers.BatchNormalization()(x2_1)
        x2_1 = layers.MaxPool2D(pool_size=(2, 2), padding="same", strides=(2, 2))(x2_1)
        x2_2 = layers.Conv2D(filters=256, kernel_size=(1, 1), strides=(2, 2), padding="same", activation="relu")(x1_2)
        x2_2 = layers.BatchNormalization()(x2_2)
        x2 = layers.Add()([x2_1, x2_2])

        x3_1 = layers.SeparableConv2D(filters=728, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(x2)
        x3_1 = layers.BatchNormalization()(x3_1)
        x3_1 = layers.SeparableConv2D(filters=728, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(x3_1)
        x3_1 = layers.BatchNormalization()(x3_1)
        x3_1 = layers.MaxPool2D(pool_size=(2, 2), padding="same", strides=(2, 2))(x3_1)
        x3_2 = layers.Conv2D(filters=728, kernel_size=(1, 1), strides=(2, 2), padding="same", activation="relu")(x2_2)
        x3_2 = layers.BatchNormalization()(x3_2)
        x3 = layers.Add()([x3_1, x3_2])

        outputs = x3
        return outputs

    @staticmethod
    def middle_flow(inputs):
        """
        xception 网络的 middle flow 模块
        :param inputs: 输入张量
        :return: 输出张量
        """
        x = inputs
        for _ in range(0, 8):
            x = layers.SeparableConv2D(filters=728, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(x)
            x = layers.BatchNormalization()(x)
            x = layers.SeparableConv2D(filters=728, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(x)
            x = layers.BatchNormalization()(x)
            x = layers.SeparableConv2D(filters=728, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(x)
            x = layers.BatchNormalization()(x)
        outputs = x
        return outputs

    @staticmethod
    def exit_flow(inputs):
        """
        xception 网络的 exit flow 模块
        :param inputs: 输入张量
        :return: 输出张量
        """
        x1_1 = layers.SeparableConv2D(filters=728, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(inputs)
        x1_1 = layers.BatchNormalization()(x1_1)
        x1_1 = layers.SeparableConv2D(filters=1024, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(x1_1)
        x1_1 = layers.BatchNormalization()(x1_1)
        x1_1 = layers.MaxPool2D(pool_size=(2, 2), padding="same", strides=(2, 2))(x1_1)
        x1_2 = layers.Conv2D(filters=1024, kernel_size=(1, 1), strides=(2, 2), padding="same", activation="relu")(inputs)
        x1_2 = layers.BatchNormalization()(x1_2)
        x = layers.Add()([x1_1, x1_2])
        x = layers.SeparableConv2D(filters=1536, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.SeparableConv2D(filters=2048, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.AveragePooling2D(pool_size=(8, 8), padding="same", strides=(1, 1))(x)
        outputs = x
        return outputs

    def xception_net(self):
        """
        xception 网络模型
        :return: 网络模型
        """
        inputs = Input(shape=self.image_shape)
        x = self.entry_flow(inputs=inputs)
        x = self.middle_flow(inputs=x)
        x = self.exit_flow(inputs=x)
        x = layers.Flatten()(x)
        outputs = layers.Dense(self.kind_of_classes, activation="softmax")(x)
        model = Model(inputs=inputs, outputs=outputs)

        return model
