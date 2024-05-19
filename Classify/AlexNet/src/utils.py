""" 绘制曲线等功能函数"""
import os
import random
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model


def plot_train_val_process_curve(history):
    """
    绘出模型训练过程中损失函数和准确率的变化过程
    :return:
    """
    accuracy = history.history["Accuracy"]
    val_accuracy = history.history["val_Accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    # plt.ion()
    plt.figure(figsize=(2, 2))
    plt.subplot(1, 2, 1)
    plt.plot(range(len(accuracy)), accuracy, label="Training Accuracy")
    plt.plot(range(len(accuracy)), val_accuracy, label="Validation Accuracy")
    plt.legend(loc="lower right")
    plt.title("Training and Validation Accuracy")
    plt.subplot(1, 2, 2)
    plt.plot(range(len(accuracy)), loss, label="Training Loss")
    plt.plot(range(len(accuracy)), val_loss, label="Validation Loss")
    plt.legend(loc="upper right")
    plt.title("Training and Validation Loss")
    plt.show(block=False)
    plt.pause(20)
    plt.close("all")


def image_predict(model_dir, test_image_dir, label_dict, image_shape):
    """
    加载模型，对图像进行预测并展示预测解结果
    :return: 对图像的预测结果
    """
    show_images = []
    model = load_model(model_dir)
    if len(os.listdir(test_image_dir)) <= 4:
        to_predict_images = os.listdir(test_image_dir)
    else:
        to_predict_images = random.sample(os.listdir(test_image_dir), 4)
    for image in to_predict_images:
        image_name = os.path.join(test_image_dir, image)
        image = cv.imread(image_name)
        image = image / 255.
        image1 = cv.resize(image, dsize=(image_shape[0], image_shape[1]))
        image = np.expand_dims(image1, axis=0)
        image_class = np.argmax(model.predict(image), axis=1)
        image = np.squeeze(image, axis=0)
        image_class = [k for k, v in label_dict.items() if v == image_class]
        image_class = ''.join(image_class)
        cv.putText(image, text=image_class, fontScale=1, thickness=2, color=(0, 255, 0), fontFace=0, org=(0, image_shape[0] - 10))
        show_images.append(image)
    horizontal_images_one = np.hstack([show_images[0], show_images[1]])
    horizontal_images_two = np.hstack([show_images[2], show_images[3]])
    vertical_images = np.vstack([horizontal_images_one, horizontal_images_two])
    cv.namedWindow(winname="image_prediction_result", flags=cv.WINDOW_AUTOSIZE)
    cv.imshow(winname="image_prediction_result", mat=vertical_images)
    cv.waitKey(0)
    cv.destroyWindow(winname="image_prediction_result")
