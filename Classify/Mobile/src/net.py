"""
说明：
    搭建VGG网络
"""
from tensorflow.keras import Input
from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras import activations


class MOBILE:
    def __init__(self, number_of_classes, image_shape):
        """
        对网络进行初始化
        :param number_of_classes: 数据集识别对象种类
        """
        self.kind_of_classes = number_of_classes
        self.image_shape = image_shape

    """用于mobile net v1 模型的各个组件"""

    @staticmethod
    def conv_block(filter_1, filter_2, strides_1, strides_2, in_tensor):
        """
        mobile net v1 基本卷积块
        :param filter_1: feature map 数量
        :param filter_2: feature map 数量
        :param strides_1: stride 步幅
        :param strides_2: stride 步幅
        :param in_tensor: 输入张量
        :return: 输出张量
        """
        x = in_tensor
        x = layers.SeparableConv2D(filters=filter_1, kernel_size=(3, 3), strides=strides_1, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filters=filter_2, kernel_size=(1, 1), strides=strides_2, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)

        out_tensor = x

        return out_tensor

    """用于mobile net v1 模型的各个组件"""

    """用于mobile net v2 模型的各个组件"""

    @staticmethod
    def bottle_neck_block_v2(t, c, n, s, in_tensor):
        """
        构建 mobile net v2 的 bottle 模块
        :param t: feature map 扩展因子，具体含义集数值参见论文
        :param c: feature map 数量
        :param n: bottle 模块的循环次数
        :param s: stride 步幅
        :param in_tensor: 输入张量
        :return: 输出张量
        """
        for _ in range(n):
            if s == 1:
                x = layers.Conv2D(filters=in_tensor.shape[-1] * t, kernel_size=(1, 1), strides=(1, 1), padding="same", activation="relu6")(in_tensor)
                x = layers.BatchNormalization()(x)
                x = layers.DepthwiseConv2D(kernel_size=(3, 3), strides=(s, s), padding="same", activation="relu6")(x)
                x = layers.BatchNormalization()(x)
                x = layers.Conv2D(filters=c, kernel_size=(1, 1), strides=(1, 1), padding="same")(x)
                x = layers.BatchNormalization()(x)
                in_tensor = layers.Conv2D(filters=c, kernel_size=(1, 1), strides=(1, 1), padding="same")(in_tensor)
                in_tensor = layers.BatchNormalization()(in_tensor)
                x = layers.Add()([x, in_tensor])
                out_tensor = x
                return out_tensor
            elif s == 2:
                x = layers.Conv2D(filters=in_tensor.shape[-1] * t, kernel_size=(1, 1), strides=(1, 1), padding="same", activation="relu6")(in_tensor)
                x = layers.BatchNormalization()(x)
                x = layers.DepthwiseConv2D(kernel_size=(3, 3), strides=(s, s), padding="same", activation="relu6")(x)
                x = layers.BatchNormalization()(x)
                x = layers.Conv2D(filters=c, kernel_size=(1, 1), strides=(1, 1), padding="same")(x)
                x = layers.BatchNormalization()(x)
                out_tensor = x
                return out_tensor
            else:
                print("please enter the right stride,it can be 1 or 2")
                pass

    @staticmethod
    def relu6(input_tensor):
        """
        relu6 激活函数
        :param input_tensor: 输入张量
        :return: 经过激活函数处理后的输出张量
        """
        x = activations.relu(x=input_tensor, max_value=6)
        return x

    """用于mobile net v2 模型的各个组件"""

    """用于mobile net v3 模型的各个组件"""

    def bottle_neck_block_v3(self, filters, expansion, using_se_block, activation_func, stride, in_tensor):
        """

        :param filters: feature map 数量，具体数值参见论文
        :param expansion: feature map 数量，具体数值参见论文
        :param using_se_block: bool值，用于判断是否使用se注意力模块
        :param activation_func: 激活函数
        :param stride: stride 步幅
        :param in_tensor: 输入张量
        :return: 输出张量
        """
        x = in_tensor
        x = layers.Conv2D(filters=expansion, kernel_size=(1, 1), strides=(1, 1), padding="same", )(x)
        x = layers.BatchNormalization()(x)
        if activation_func == "hs":
            x = self.h_swish(x)
        elif activation_func == "re":
            x = layers.Activation("relu")(x)
        else:
            x = x
        x = layers.DepthwiseConv2D(kernel_size=(3, 3), strides=stride, padding="same")(x)
        x = layers.BatchNormalization()(x)
        if activation_func == "hs":
            x = self.h_swish(x)
        elif activation_func == "re":
            x = layers.Activation("relu")(x)
        else:
            x = x
        if using_se_block:
            x = self.se_block(reduction=16, in_tensor=x)
        x = layers.Conv2D(filters=filters, kernel_size=(1, 1), strides=(1, 1), padding="same")(x)
        x = layers.BatchNormalization()(x)
        if stride == (1, 1) and in_tensor.shape[-1] == x.shape[-1]:
            x = layers.Add()([in_tensor, x])

        outputs = x
        return outputs

    @staticmethod
    def se_block(reduction, in_tensor):
        """
        se注意力模块，具体原理参见论文。
        :param reduction: 缩减因子，代码使用论文默认数值16
        :param in_tensor: 输入张量
        :return: 输出张量
        """
        x = layers.GlobalAvgPool2D()(in_tensor)
        x = layers.Dense(units=in_tensor.shape[-1] / reduction, activation="relu")(x)
        x = layers.Dense(units=in_tensor.shape[-1], activation="sigmoid")(x)
        x = layers.Reshape((1, 1, in_tensor.shape[-1]))(x)
        out_puts = layers.Multiply()([x, in_tensor])
        return out_puts

    @staticmethod
    def h_swish(input_tensor):
        """
        论文所提出的 h_swish 激活函数
        :param input_tensor: 输入张量
        :return: 经过激活函数处理后的输出张量
        """
        x = activations.relu(x=(input_tensor + 3), max_value=6)
        x = input_tensor * x / 6

        return x

    def mobile_net(self, version):
        """
        mobile net v1 v2 v3 模型
        :param version: 选择所需要的版本
        :return: 网络模型
        """
        if version == 1:
            inputs = Input(shape=self.image_shape)
            x = layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), padding="same", activation="relu")(inputs)
            x = self.conv_block(filter_1=32, filter_2=64, strides_1=(1, 1), strides_2=(1, 1), in_tensor=x)
            x = self.conv_block(filter_1=64, filter_2=128, strides_1=(2, 2), strides_2=(1, 1), in_tensor=x)
            x = self.conv_block(filter_1=128, filter_2=128, strides_1=(1, 1), strides_2=(1, 1), in_tensor=x)
            x = self.conv_block(filter_1=128, filter_2=256, strides_1=(2, 2), strides_2=(1, 1), in_tensor=x)
            x = self.conv_block(filter_1=256, filter_2=256, strides_1=(1, 1), strides_2=(1, 1), in_tensor=x)
            x = self.conv_block(filter_1=256, filter_2=512, strides_1=(2, 2), strides_2=(1, 1), in_tensor=x)
            for _ in range(5):
                x = self.conv_block(filter_1=512, filter_2=512, strides_1=(1, 1), strides_2=(1, 1), in_tensor=x)
            x = self.conv_block(filter_1=512, filter_2=1024, strides_1=(2, 2), strides_2=(1, 1), in_tensor=x)
            x = self.conv_block(filter_1=1024, filter_2=1024, strides_1=(2, 2), strides_2=(1, 1), in_tensor=x)
            x = layers.AveragePooling2D(pool_size=(7, 7), strides=(1, 1), padding="same")(x)
            flatten = layers.Flatten()(x)
            outputs = layers.Dense(units=self.kind_of_classes, activation="softmax")(flatten)

            model = Model(inputs=inputs, outputs=outputs)

            return model
        elif version == 2:
            inputs = Input(shape=self.image_shape)
            x = layers.Conv2D(filters=32, kernel_size=(1, 1), strides=(2, 2), padding="same", activation="relu")(inputs)
            x = layers.BatchNormalization()(x)
            x = self.bottle_neck_block_v2(t=1, c=16, n=1, s=1, in_tensor=x)
            x = self.bottle_neck_block_v2(t=6, c=24, n=2, s=2, in_tensor=x)
            x = self.bottle_neck_block_v2(t=6, c=32, n=3, s=2, in_tensor=x)
            x = self.bottle_neck_block_v2(t=6, c=64, n=4, s=2, in_tensor=x)
            x = self.bottle_neck_block_v2(t=6, c=96, n=3, s=1, in_tensor=x)
            x = self.bottle_neck_block_v2(t=6, c=160, n=3, s=2, in_tensor=x)
            x = self.bottle_neck_block_v2(t=6, c=320, n=1, s=1, in_tensor=x)
            x = layers.Conv2D(filters=1280, kernel_size=(1, 1), strides=(1, 1), padding="same", activation="relu")(x)
            x = layers.AveragePooling2D(pool_size=(7, 7), padding="same", strides=(1, 1))(x)
            x = layers.Flatten()(x)
            x = layers.Dense(units=self.kind_of_classes, activation="softmax")(x)

            outputs = x
            model = Model(inputs=inputs, outputs=outputs)

            return model

        elif version == 3:
            inputs = Input(shape=self.image_shape)
            x = layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(2, 2), padding="same", activation="relu")(inputs)
            x = layers.BatchNormalization()(x)
            x = self.bottle_neck_block_v3(filters=16, expansion=16, using_se_block=False, activation_func="re", stride=(1, 1), in_tensor=x)
            x = self.bottle_neck_block_v3(filters=24, expansion=64, using_se_block=False, activation_func="re", stride=(2, 2), in_tensor=x)
            x = self.bottle_neck_block_v3(filters=24, expansion=72, using_se_block=False, activation_func="re", stride=(1, 1), in_tensor=x)
            x = self.bottle_neck_block_v3(filters=40, expansion=72, using_se_block=True, activation_func="re", stride=(2, 2), in_tensor=x)
            x = self.bottle_neck_block_v3(filters=40, expansion=120, using_se_block=True, activation_func="re", stride=(1, 1), in_tensor=x)
            x = self.bottle_neck_block_v3(filters=40, expansion=120, using_se_block=True, activation_func="re", stride=(1, 1), in_tensor=x)
            x = self.bottle_neck_block_v3(filters=80, expansion=240, using_se_block=False, activation_func="hs", stride=(2, 2), in_tensor=x)
            x = self.bottle_neck_block_v3(filters=80, expansion=200, using_se_block=False, activation_func="hs", stride=(1, 1), in_tensor=x)
            x = self.bottle_neck_block_v3(filters=80, expansion=184, using_se_block=False, activation_func="hs", stride=(1, 1), in_tensor=x)
            x = self.bottle_neck_block_v3(filters=80, expansion=184, using_se_block=False, activation_func="hs", stride=(1, 1), in_tensor=x)
            x = self.bottle_neck_block_v3(filters=112, expansion=480, using_se_block=True, activation_func="hs", stride=(1, 1), in_tensor=x)
            x = self.bottle_neck_block_v3(filters=112, expansion=672, using_se_block=True, activation_func="hs", stride=(1, 1), in_tensor=x)
            x = self.bottle_neck_block_v3(filters=160, expansion=672, using_se_block=True, activation_func="hs", stride=(2, 2), in_tensor=x)
            x = self.bottle_neck_block_v3(filters=160, expansion=960, using_se_block=True, activation_func="hs", stride=(1, 1), in_tensor=x)
            x = self.bottle_neck_block_v3(filters=160, expansion=960, using_se_block=True, activation_func="hs", stride=(1, 1), in_tensor=x)
            x = layers.Conv2D(filters=960, kernel_size=(1, 1), strides=(1, 1), padding="same")(x)
            x = self.h_swish(x)
            x = layers.BatchNormalization()(x)
            x = layers.AveragePooling2D(pool_size=(7, 7), strides=(1, 1), padding="same")(x)
            x = layers.Conv2D(filters=1280, kernel_size=(1, 1), strides=(1, 1), padding="same")(x)
            x = self.h_swish(x)
            x = layers.Conv2D(filters=self.kind_of_classes, kernel_size=(1, 1), strides=(1, 1), padding="same", activation="relu")(x)
            x = layers.Flatten()(x)
            x = layers.Dense(units=self.kind_of_classes, activation="softmax")(x)

            outputs = x
            model = Model(inputs=inputs, outputs=outputs)

            return model

        else:
            print("please enter 1 or 2 or 3")
            pass
