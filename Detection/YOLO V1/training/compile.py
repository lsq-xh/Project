"""
说明：
    对模型进行编译
"""
import os
from training.loss import Loss
from training.net import YOLO
from tensorflow.keras import callbacks
from tensorflow.keras import optimizers


class CompileModel:
    def __init__(self):
        pass

    @staticmethod
    def callback_functions(training_log_dir):
        """
        一系列回调函数
        :return: 一系列回调函数
        """
        terminate_on_nan = callbacks.TerminateOnNaN()
        early_stopping = callbacks.EarlyStopping(monitor="val_loss", min_delta=1e-5, patience=5, mode="auto")
        reduce_lr_on_plateau = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=0, mode='min', min_delta=0.001)
        checkpoint = callbacks.ModelCheckpoint(filepath=training_log_dir, monitor="val_loss", mode="min", save_best_only=True, save_weights_only=False, save_freq='epoch')
        # self.lr_scheduler = callbacks.LearningRateScheduler(self.scheduler())
        # self.tensorBoard = callbacks.TensorBoard()

        return early_stopping, reduce_lr_on_plateau, checkpoint, terminate_on_nan,

    """
    def scheduler(self):
        回调函数LearningRateScheduler所需要传入参数，可自由选择
        :return: 更改后的学习率

        if self.epochs < 10:
            return self.learning_rate
        else:
            return self.learning_rate * tf.math.exp(-0.1)
    """

    def call_model(self, train_dataset, val_dataset, test_dataset, epochs, learning_rate, training_log_dir, model_save_dir, only_evaluate, input_image_shape, out_shape, numer_of_object_classes):
        """
        对模型进行训练和评估
        :return: 模型训练history记录
        """
        model = YOLO(input_image_shape=input_image_shape, output_tensor_shape=out_shape).yolo_net()
        # model.summary()
        if not only_evaluate:
            callbacks_list = self.callback_functions(training_log_dir)
            model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.99),
                          loss=Loss(number_of_grid_box=2, image_shape=(448, 448, 3), lambda_coord=5, lambda_none_object=0.5, numer_of_object_classes=numer_of_object_classes).compute_yolo_loss)
            history = model.fit(x=train_dataset, epochs=epochs, validation_data=val_dataset, callbacks=callbacks_list)
            model.save(os.path.join(model_save_dir, "yolo_v1.h5"), save_format="tf")
            model.evaluate(test_dataset)
            return history
        else:
            model.evaluate(test_dataset)
