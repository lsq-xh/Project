import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # 防止输出过多的tensorflow日志信息
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
base_image_dir = r'E:/CV/DataSets/Object_detection_VOC2007/'
training_log_dir = r"E:/CV/Models/YOLO V1/train_log/"
model_save_dir = r"E:/CV/Models/YOLO V1/model_save/"
sample_image_dir = r"E:/CV/Models/YOLO V1/samples"

image_shape = (448, 448, 3)
batch_size = 48
epochs = 150
lr = 1e-4
numer_of_box = 2
grid_size = [7, 7]
object_class = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
output_shape = (5 * numer_of_box + len(object_class)) * grid_size[0] * grid_size[1]
