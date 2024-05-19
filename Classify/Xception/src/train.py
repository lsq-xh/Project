"""
说明：
    对模型进行编译、训练、评估
"""
import os
from tensorflow.keras import callbacks
from tensorflow.keras import optimizers


class TrainEValuateModel:
    def __init__(self, epochs, learning_rate, training_log_dir, model_save_dir, dataset, net):
        """
        模型训练评估初始化参数
        :param epochs: 网络训练次数
        :param learning_rate: 学习率
        :param training_log_dir: 训练记录(ModelCheckpoint)存放路径
        :param model_save_dir: 模型保存路径
        """
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.training_log_dir = training_log_dir
        self.model_save_dir = model_save_dir
        self.ds = dataset
        self.model = net

    def callback_functions(self):
        """
        一系列回调函数
        :return: 一系列回调函数
        """
        terminate_on_nan = callbacks.TerminateOnNaN()
        early_stopping = callbacks.EarlyStopping(monitor="val_loss", min_delta=1e-5, patience=5, mode="auto")
        reduce_lr_on_plateau = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=0, mode='min', min_delta=0.001)
        checkpoint = callbacks.ModelCheckpoint(filepath=self.training_log_dir, monitor="val_loss", mode="min", save_best_only=True, save_weights_only=False, save_freq='epoch')
        # self.lr_scheduler = callbacks.LearningRateScheduler(self.scheduler())
        # self.tensorBoard = callbacks.TensorBoard()

        return early_stopping, checkpoint, terminate_on_nan, reduce_lr_on_plateau

    # def scheduler(self):
    #
    #     """
    #     回调函数LearningRateScheduler所需要传入参数，可自由选择
    #     :return: 更改后的学习率
    #     """
    #     if self.epochs < 10:
    #         return self.learning_rate
    #     else:
    #         return self.learning_rate * tf.math.exp(-0.1)

    def train_and_evaluate_model(self):
        """
        对模型进行训练和评估
        :return: 模型训练history记录
        """
        train_ds = self.ds.get_train_dataset()
        val_ds = self.ds.get_val_dataset()
        test_ds = self.ds.get_test_dataset()
        print(train_ds)
        callbacks_list = self.callback_functions()
        self.model.compile(optimizer=optimizers.RMSprop(learning_rate=self.learning_rate), loss='categorical_crossentropy', metrics=["Accuracy"])
        self.model.summary()
        history = self.model.fit(x=train_ds, epochs=self.epochs, validation_data=val_ds, callbacks=callbacks_list)
        self.model.save(os.path.join(self.model_save_dir, "Xception.h5"), save_format="tf")
        self.model.evaluate(test_ds)
        return history
