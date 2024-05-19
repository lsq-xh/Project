"""
说明：
    搭建VGG网络
"""
from tensorflow.keras import Input
from tensorflow.keras import Model
from tensorflow.keras import layers


class RES:
    def __init__(self, number_of_classes, image_shape, res_layers):
        """
        网络参数初始化
        :param number_of_classes: 分类图像种类
        :param image_shape: 图像形状
        :param res_layers: 要选择的res网络的层数
        """
        self.kind_of_classes = number_of_classes
        self.image_shape = image_shape
        self.res_layers = res_layers

    def residual_block(self, filters, in_tensor, bool_flag):
        """
        构建不同层数的残差块
        :param filters: filter的数量
        :param in_tensor: 输入张量
        :param bool_flag: 残差连接的形式进行判断
        :return:
        """
        if self.res_layers == 18 or self.res_layers == 34:
            if bool_flag:
                x = layers.Conv2D(filters=filters, kernel_size=(3, 3), strides=(2, 2), padding="same", activation="relu")(in_tensor)
                x = layers.BatchNormalization()(x)
                x = layers.Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(x)
                x = layers.BatchNormalization()(x)
                in_tensor_branch = layers.Conv2D(filters=filters, kernel_size=(1, 1), strides=(2, 2), padding="same", activation="relu")(in_tensor)
                in_tensor_branch = layers.BatchNormalization()(in_tensor_branch)
                out_tensor = layers.Add()([in_tensor_branch, x])
                return out_tensor
            else:
                x = layers.Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(in_tensor)
                x = layers.BatchNormalization()(x)
                x = layers.Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(x)
                x = layers.BatchNormalization()(x)
                out_tensor = layers.Add()([in_tensor, x])
                return out_tensor
        elif self.res_layers == 50 or self.res_layers == 101 or self.res_layers == 152:
            x = layers.Conv2D(filters=filters, kernel_size=(1, 1), strides=(1, 1), padding="same", activation="relu")(in_tensor)
            x = layers.BatchNormalization()(x)
            if bool_flag:
                if filters == 64:
                    x = layers.Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(x)
                    x = layers.BatchNormalization()(x)
                else:
                    x = layers.Conv2D(filters=filters, kernel_size=(3, 3), strides=(2, 2), padding="same", activation="relu")(x)
                    x = layers.BatchNormalization()(x)
                x = layers.Conv2D(filters=filters * 4, kernel_size=(1, 1), strides=(1, 1), padding="same", activation="relu")(x)
                x = layers.BatchNormalization()(x)
                if filters == 64:
                    in_tensor_branch = layers.Conv2D(filters=filters * 4, kernel_size=(1, 1), strides=(1, 1), padding="same", activation="relu")(in_tensor)
                    in_tensor_branch = layers.BatchNormalization()(in_tensor_branch)
                else:
                    in_tensor_branch = layers.Conv2D(filters=filters * 4, kernel_size=(1, 1), strides=(2, 2), padding="same", activation="relu")(in_tensor)
                    in_tensor_branch = layers.BatchNormalization()(in_tensor_branch)
            else:
                x = layers.Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(x)
                x = layers.BatchNormalization()(x)
                x = layers.Conv2D(filters=filters * 4, kernel_size=(1, 1), strides=(1, 1), padding="same", activation="relu")(x)
                x = layers.BatchNormalization()(x)
                in_tensor_branch = in_tensor
            out_tensor = layers.Add()([x, in_tensor_branch])
            return out_tensor
        else:
            print("no more model layer, please enter number 18、34、50、101 or 152")
            pass

    def res_net(self):
        """
        搭建不同层数的残差网络
        :return: 网络输出张量
        """
        inputs = Input(shape=self.image_shape)
        x = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding="same", activation="relu")(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same")(x)
        if self.res_layers == 18:
            x = self.residual_block(filters=64, in_tensor=x, bool_flag=False)
            x = self.residual_block(filters=64, in_tensor=x, bool_flag=False)
            x = self.residual_block(filters=128, in_tensor=x, bool_flag=True)
            x = self.residual_block(filters=128, in_tensor=x, bool_flag=False)
            x = self.residual_block(filters=256, in_tensor=x, bool_flag=True)
            x = self.residual_block(filters=256, in_tensor=x, bool_flag=False)
            x = self.residual_block(filters=512, in_tensor=x, bool_flag=True)
            x = self.residual_block(filters=512, in_tensor=x, bool_flag=False)
        elif self.res_layers == 34:
            for _ in range(3):
                x = self.residual_block(filters=64, in_tensor=x, bool_flag=False)
            x = self.residual_block(filters=128, in_tensor=x, bool_flag=True)
            for _ in range(3):
                x = self.residual_block(filters=128, in_tensor=x, bool_flag=False)
            x = self.residual_block(filters=256, in_tensor=x, bool_flag=True)
            for _ in range(5):
                x = self.residual_block(filters=256, in_tensor=x, bool_flag=False)
            x = self.residual_block(filters=512, in_tensor=x, bool_flag=True)
            for _ in range(2):
                x = self.residual_block(filters=512, in_tensor=x, bool_flag=False)
        elif self.res_layers == 50:
            x = self.residual_block(filters=64, in_tensor=x, bool_flag=True)
            for _ in range(2):
                self.residual_block(filters=64, in_tensor=x, bool_flag=False)
            x = self.residual_block(filters=128, in_tensor=x, bool_flag=True)
            for _ in range(3):
                self.residual_block(filters=128, in_tensor=x, bool_flag=False)
            x = self.residual_block(filters=256, in_tensor=x, bool_flag=True)
            for _ in range(5):
                self.residual_block(filters=256, in_tensor=x, bool_flag=False)
            x = self.residual_block(filters=512, in_tensor=x, bool_flag=True)
            for _ in range(2):
                self.residual_block(filters=512, in_tensor=x, bool_flag=False)
        elif self.res_layers == 101:
            x = self.residual_block(filters=64, in_tensor=x, bool_flag=True)
            for _ in range(2):
                self.residual_block(filters=64, in_tensor=x, bool_flag=False)
            x = self.residual_block(filters=128, in_tensor=x, bool_flag=True)
            for _ in range(3):
                self.residual_block(filters=128, in_tensor=x, bool_flag=False)
            x = self.residual_block(filters=256, in_tensor=x, bool_flag=True)
            for _ in range(22):
                self.residual_block(filters=256, in_tensor=x, bool_flag=False)
            x = self.residual_block(filters=512, in_tensor=x, bool_flag=True)
            for _ in range(2):
                self.residual_block(filters=512, in_tensor=x, bool_flag=False)
        elif self.res_layers == 152:
            x = self.residual_block(filters=64, in_tensor=x, bool_flag=True)
            for _ in range(2):
                self.residual_block(filters=64, in_tensor=x, bool_flag=False)
            x = self.residual_block(filters=128, in_tensor=x, bool_flag=True)
            for _ in range(7):
                self.residual_block(filters=128, in_tensor=x, bool_flag=False)
            x = self.residual_block(filters=256, in_tensor=x, bool_flag=True)
            for _ in range(35):
                self.residual_block(filters=256, in_tensor=x, bool_flag=False)
            x = self.residual_block(filters=512, in_tensor=x, bool_flag=True)
            for _ in range(2):
                self.residual_block(filters=512, in_tensor=x, bool_flag=False)
        else:
            print("no more model layer, please enter number 18、34、50、101 or 152")
            pass
        x = layers.AvgPool2D(pool_size=(7, 7), strides=(1, 1))(x)
        x = layers.Flatten()(x)
        outputs = layers.Dense(self.kind_of_classes, activation="softmax")(x)
        model = Model(inputs=inputs, outputs=outputs)
        return model
