import os
import tensorflow as tf
from data import DataProcess
from net import RES
from utils import plot_train_val_process_curve
from utils import image_predict
from train import TrainEValuateModel


def main():
    """
    程序主函数
    :return:
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"   # 使用GPU 0进行训练
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # 防止输出过多的tensorflow日志信息
    data_dir = r"E:/DataSets/Service Hall classify/"
    training_log_dir = r"E:/Code_Py/Res/train_log/"
    test_image_dir = r"E:/Code_Py/Res/test_image/"
    model_save_dir = r"E:/Code_Py/Res/model_save/"
    image_shape = (224, 224, 3)
    batch_size = 80
    epochs = 300
    lr = 1e-4

    datasets = DataProcess(file_path=data_dir, image_shape=image_shape, batch_size=batch_size)
    model = RES(number_of_classes=datasets.get_object_class_and_number()[0], image_shape=image_shape, res_layers=34).res_net()
    print(datasets.get_object_class_and_number()[0])
    label_dict = datasets.get_label_dict()
    if not os.path.exists(model_save_dir):
        if tf.config.list_physical_devices('GPU'):
            print("training by GPU")
            train_evaluate_model = TrainEValuateModel(dataset=datasets, net=model, epochs=epochs, learning_rate=lr,
                                                      model_save_dir=model_save_dir, training_log_dir=training_log_dir)
            train_history = train_evaluate_model.train_and_evaluate_model()
            plot_train_val_process_curve(history=train_history)
            image_predict(model_dir=os.path.join(model_save_dir, "res.h5"), test_image_dir=test_image_dir, label_dict=label_dict, image_shape=image_shape)
        else:
            print("training by CPU")
            pass
    else:
        print("model exits, do not train again")
        image_predict(model_dir=os.path.join(model_save_dir, "res.h5"), test_image_dir=test_image_dir, label_dict=label_dict, image_shape=image_shape)


if __name__ == "__main__":
    main()