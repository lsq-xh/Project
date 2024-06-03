import tensorflow as tf
from utils.config import *
from utils.show import Show
from data.loader import DataLoader
from training.compile import CompileModel


def main():
    """
    程序主函数
    :return:
    """
    loader = DataLoader(file_base_dir=base_image_dir, batch_size=batch_size, list_of_class_name_of_object=object_class)
    image_dataset = loader.get_image_dataset_from_image_set_direct(train_val_test_ration=[0.7, 0.15, 0.15])
    train_dataset = loader.get_train_dataset(image_shape=(448, 448, 3), datasets=image_dataset[0])
    val_dataset = loader.get_val_dataset(image_shape=(448, 448, 3), datasets=image_dataset[1])
    test_dataset = loader.get_test_dataset(image_shape=(448, 448, 3), datasets=image_dataset[2])
    if not os.path.exists(model_save_dir):
        if tf.config.list_physical_devices('GPU'):
            print("training by GPU")
            model = CompileModel()
            history = model.call_model(train_dataset=train_dataset, val_dataset=val_dataset, test_dataset=test_dataset, epochs=epochs, learning_rate=lr,
                                       training_log_dir=training_log_dir, model_save_dir=model_save_dir, only_evaluate=False, input_image_shape=image_shape, out_shape=output_shape,
                                       numer_of_object_classes=len(object_class))
            Show().show_train_curve(history=history)
            Show().show_image_ground_truth_result(file_path=sample_image_dir, image_shape=(image_shape[0], image_shape[1]))
        else:
            print("GPU is not available, check your environment")
            pass


if __name__ == "__main__":
    main()
