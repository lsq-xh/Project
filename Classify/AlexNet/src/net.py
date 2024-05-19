"""
说明：
    搭建VGG网络
"""
from tensorflow.keras import Input
from tensorflow.keras import Model
from tensorflow.keras import layers


class Alex:
    def __init__(self, number_of_classes, image_shape):
        """
        对网络进行初始化
        :param number_of_classes: 数据集识别对象种类
        """
        self.kind_of_classes = number_of_classes
        self.image_shape = image_shape

    def alex_net(self):
        """
        构建Alex网络
        :return:
        """
        inputs = Input(shape=self.image_shape)
        x = layers.Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), padding="same", activation="relu")(inputs)
        x = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="same")(x)
        x = layers.Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), padding="same", activation="relu")(x)
        x = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="same")(x)
        x = layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(x)
        x = layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(x)
        x = layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(x)
        x = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="same")(x)
        x = layers.Flatten()(x)
        x = layers.Dense(units=40)(x)
        x = layers.Dropout(0.8)(x)
        x = layers.Dense(units=40)(x)
        x = layers.Dropout(0.8)(x)
        outputs = layers.Dense(units=self.kind_of_classes, activation="softmax")(x)

        model = Model(inputs=inputs, outputs=outputs)

        return model
