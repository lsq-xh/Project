"""
说明：
    搭建VGG网络
"""
from tensorflow.keras import Input
from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras import regularizers


class VGG:
    def __init__(self, number_of_classes, image_shape):
        """
        对网络进行初始化
        :param number_of_classes: 数据集识别对象种类
        """
        self.kind_of_classes = number_of_classes
        self.image_shape = image_shape

    def vgg_net(self):
        """
        vgg 网络模型搭建
        :return: 网络模型
        """
        inputs = Input(shape=self.image_shape)
        conv1_1 = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(inputs)
        conv1_2 = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(conv1_1)
        pool1_1 = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same")(conv1_2)

        conv2_1 = layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(pool1_1)
        conv2_2 = layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(conv2_1)
        pool2_1 = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same")(conv2_2)

        conv3_1 = layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(pool2_1)
        conv3_2 = layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(conv3_1)
        conv3_3 = layers.Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding="same", activation="relu")(conv3_2)
        pool3_1 = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same")(conv3_3)

        conv4_1 = layers.Conv2D(filters=512, kernel_size=(2, 2), strides=(1, 1), padding="same", activation="relu")(pool3_1)
        conv4_2 = layers.Conv2D(filters=512, kernel_size=(2, 2), strides=(1, 1), padding="same", activation="relu")(conv4_1)
        conv4_3 = layers.Conv2D(filters=512, kernel_size=(1, 1), strides=(1, 1), padding="same", activation="relu")(conv4_2)
        pool4_1 = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same")(conv4_3)

        conv5_1 = layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(pool4_1)
        conv5_2 = layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(conv5_1)
        conv5_3 = layers.Conv2D(filters=512, kernel_size=(1, 1), strides=(1, 1), padding="same", activation="relu")(conv5_2)
        pool5_1 = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same")(conv5_3)
        flatten = layers.Flatten()(pool5_1)
        dense_1 = layers.Dense(units=40, activation="relu")(flatten)
        drop_1 = layers.Dropout(rate=0.8)(dense_1)
        dense_2 = layers.Dense(units=40, activation="relu")(drop_1)
        drop_2 = layers.Dropout(rate=0.8)(dense_2)
        outputs = layers.Dense(units=self.kind_of_classes, activation="softmax")(flatten)
        model = Model(inputs=inputs, outputs=outputs)

        return model
