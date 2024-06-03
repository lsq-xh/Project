"""
说明：
    搭建SSD网络
"""
from tensorflow.keras import Input
from tensorflow.keras import Model
from tensorflow.keras import layers


class YOLO:
    def __init__(self, input_image_shape, output_tensor_shape):
        """
        初始化 YOLO 网络参数
        :param output_tensor_shape: 网络输出张量的形状
        """
        self.output_tensor_shape = output_tensor_shape
        self.input_image_shape = input_image_shape

    @staticmethod
    def conv_block(input_tensor, n_filters, kernel_size, strides, pool=False):
        """
        定义卷积块
        :param input_tensor: 输入张量
        :param n_filters: 滤波器数量
        :param kernel_size: 卷积核大小
        :param strides: stride 步幅
        :param pool: 是否进行池化判断标志 ，bool型
        :return: 经过卷积后的输出张量
        """
        x = layers.Conv2D(filters=n_filters, kernel_size=kernel_size, padding='same', strides=strides, kernel_initializer='he_normal')(input_tensor)
        x = layers.LeakyReLU(alpha=0.1)(x)
        if pool:
            x = layers.MaxPool2D(pool_size=2, strides=(2, 2), padding="same")(x)
        out_tensor = x
        return out_tensor

    def yolo_net(self):
        """
        YOLO V1 完整的结构
        :return: 经过 YOLO V1 网络输出的张量
        """
        input_tensor = Input(self.input_image_shape)
        x = self.conv_block(input_tensor=input_tensor, n_filters=64, kernel_size=(7, 7), strides=(2, 2), pool=True)
        x = self.conv_block(input_tensor=x, n_filters=192, kernel_size=(3, 3), strides=(1, 1), pool=True)
        x = self.conv_block(input_tensor=x, n_filters=128, kernel_size=(1, 1), strides=(1, 1))
        x = self.conv_block(input_tensor=x, n_filters=256, kernel_size=(3, 3), strides=(1, 1))
        x = self.conv_block(input_tensor=x, n_filters=256, kernel_size=(1, 1), strides=(1, 1))
        x = self.conv_block(input_tensor=x, n_filters=512, kernel_size=(3, 3), strides=(1, 1), pool=True)
        for _ in range(4):
            x = self.conv_block(input_tensor=x, n_filters=256, kernel_size=(1, 1), strides=(1, 1))
            x = self.conv_block(input_tensor=x, n_filters=512, kernel_size=(3, 3), strides=(1, 1))
        x = self.conv_block(input_tensor=x, n_filters=512, kernel_size=(1, 1), strides=(1, 1))
        x = self.conv_block(input_tensor=x, n_filters=1024, kernel_size=(3, 3), strides=(1, 1), pool=True)
        for _ in range(2):
            x = self.conv_block(input_tensor=x, n_filters=512, kernel_size=(3, 3), strides=(1, 1))
            x = self.conv_block(input_tensor=x, n_filters=1024, kernel_size=(3, 3), strides=(1, 1))
        x = self.conv_block(input_tensor=x, n_filters=1024, kernel_size=(3, 3), strides=(1, 1))
        x = self.conv_block(input_tensor=x, n_filters=1024, kernel_size=(3, 3), strides=(1, 1), pool=True)
        x = self.conv_block(input_tensor=x, n_filters=1024, kernel_size=(3, 3), strides=(1, 1))
        x = layers.Flatten()(x)
        x = layers.Dense(units=200)(x)
        x = layers.Dropout(rate=0.5)(x)
        x = layers.Dense(units=200)(x)
        x = layers.Dropout(rate=0.5)(x)

        output_tensor = layers.Dense(units=self.output_tensor_shape)(x)

        model = Model(inputs=input_tensor, outputs=output_tensor)

        return model
