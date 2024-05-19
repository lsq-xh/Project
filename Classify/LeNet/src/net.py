"""
说明：
    搭建VGG网络
"""
from tensorflow.keras import Input
from tensorflow.keras import Model
from tensorflow.keras import layers


class Le:
    def __init__(self, number_of_classes, image_shape):
        """
        对网络进行初始化
        :param number_of_classes: 数据集识别对象种类
        """
        self.kind_of_classes = number_of_classes
        self.image_shape = image_shape

    def le_net(self):
        """
        搭建le-5 net 网络
        :return: 返回模型
        """
        inputs = Input(shape=self.image_shape)
        x = layers.Conv2D(filters=6, kernel_size=5, strides=1, activation='sigmoid')(inputs)
        x = layers.MaxPool2D(pool_size=2, strides=2)(x)
        x = layers.Conv2D(filters=16, kernel_size=5, strides=1, activation='sigmoid')(x)
        x = layers.MaxPool2D(pool_size=2, strides=2)(x)
        x = layers.Conv2D(filters=120, kernel_size=5, strides=1, activation='sigmoid')(x)
        x = layers.Flatten()(x)
        x = layers.Dense(units=84, activation='sigmoid')(x)
        outputs = layers.Dense(units=self.kind_of_classes, activation="softmax")(x)
        model = Model(inputs=inputs, outputs=outputs)

        return model
