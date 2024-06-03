""" 定义 YOLO 网络的损失函数 """
import tensorflow as tf
from utils.iou import IOU
from utils.config import grid_size


class Loss:
    def __init__(self, number_of_grid_box, image_shape, lambda_coord, lambda_none_object, numer_of_object_classes):
        """
        损失函数初始化函数
        :param number_of_grid_box: yolo 每个网格用于预测的框的个数，yolo v1 中即 B 的值，等于2
        :param image_shape: 输入网络的图像的大小
        :param lambda_coord: 坐标损失权重
        :param lambda_none_object: 损失函数权重
        """
        self.number_of_grid_box = number_of_grid_box
        self.image_shape = image_shape
        self.lambda_coord = lambda_coord
        self.lambda_none_object = lambda_none_object
        self.numer_of_object_classes = numer_of_object_classes

    def compute_yolo_loss(self, y_true, y_prediction):
        """
        计算 yolo 的损失函数
        :param y_true: 真实值标签，真实标签格式为 [confidence, x, y, w, h, class]
        :param y_prediction: 预测值标签, 其是一个 7*7*（5+20）的一维张量
        :return: 损失函数的值
        """
        y_prediction = tf.reshape(tensor=y_prediction, shape=(-1, grid_size[0], grid_size[1], 5 * self.number_of_grid_box + self.numer_of_object_classes))
        prediction_object_confidence = tf.gather(y_prediction, indices=[0, 5], axis=3)
        prediction_object_classes = y_prediction[:, :, :, 5 * self.number_of_grid_box:]
        prediction_object_box_coordinate = tf.gather(y_prediction, indices=[1, 2, 3, 4, 6, 7, 8, 9], axis=3)
        # 分别得到真实标签输出结果中的置信度、类别以及坐标
        true_object_confidence = y_true[:, :, :, :1]
        true_object_box_coordinate = y_true[:, :, :, 1:5]
        true_object_classes = y_true[:, :, :, 5:]
        # 分别重塑网络输出结果和真实标签的张量维度，其中真实标签中的坐标一项进行了扩充，重塑后的网络输出结果和真实标签维度一致,用于计算最终的损失函数
        true_object_box_coordinate = tf.reshape(true_object_box_coordinate, shape=[-1, grid_size[0], grid_size[1], 1, 4])
        true_object_box_coordinate = tf.tile(true_object_box_coordinate, multiples=[1, 1, 1, self.number_of_grid_box, 1])
        prediction_object_box_coordinate = tf.reshape(prediction_object_box_coordinate, shape=[-1, grid_size[0], grid_size[1], self.number_of_grid_box, 4])
        # 对每一个边框需要预测其（x, y, w, h），其中（x, y）是相对于负责预测的grid左上角坐标的偏移量，（w,h）是相对于整个图像的宽度和高度，因此需要构建预测值和真实值的中心点（x,y）的坐标相对于每个网格左上角的坐标系,同时将坐标转换为[x_min,y_min,x_max,y_max]的格式
        # 处理方式为对坐标进行处理，计算每个坐标在相对于网格左上角的坐标的值，范围为 0~1，计算公式为: coordinate = (coordinate * cell_size) / image_size - offset
        # 构建网格坐标，并计算出每个偏移量 offset
        grid_x_range = tf.range(grid_size[0], dtype=tf.float32)
        grid_y_range = tf.range(grid_size[1], dtype=tf.float32)
        grid_x, grid_y = tf.meshgrid(grid_x_range, grid_y_range)
        x_coordinate_offset = tf.reshape(grid_x, (-1, 1))
        y_coordinate_offset = tf.reshape(grid_y, (-1, 1))
        x_y_coordinate_offset = tf.concat([x_coordinate_offset, y_coordinate_offset], axis=-1)
        x_y_coordinate_offset = tf.cast(tf.reshape(x_y_coordinate_offset, [grid_size[0], grid_size[1], 1, 2]), tf.float32)
        true_object_box_coordinate = tf.stack(values=[(true_object_box_coordinate[..., 0] * grid_size[0] / self.image_shape[0] - x_y_coordinate_offset[..., 0]),
                                                      (true_object_box_coordinate[..., 1] * grid_size[0] / self.image_shape[0] - x_y_coordinate_offset[..., 1]),
                                                      (tf.square(true_object_box_coordinate[..., 2])),
                                                      (tf.square(true_object_box_coordinate[..., 3]))], axis=-1)
        prediction_object_box_coordinate = tf.stack(values=[(prediction_object_box_coordinate[..., 0] * grid_size[0] / self.image_shape[0] - x_y_coordinate_offset[..., 0]),
                                                            (prediction_object_box_coordinate[..., 1] * grid_size[0] / self.image_shape[0] - x_y_coordinate_offset[..., 1]),
                                                            (tf.sqrt(prediction_object_box_coordinate[..., 2])),
                                                            (tf.sqrt(prediction_object_box_coordinate[..., 3]))], axis=-1)
        # 将坐标变换为[x_min,y_min,x_max,y_max]的格式，方便计算真实框和预测框之间的IOU
        true_object_box_normalized_coordinate = tf.stack(values=[(true_object_box_coordinate[..., 0] - true_object_box_coordinate[..., 2] / 2),
                                                                 (true_object_box_coordinate[..., 1] - true_object_box_coordinate[..., 3] / 2),
                                                                 (true_object_box_coordinate[..., 0] + true_object_box_coordinate[..., 2] / 2),
                                                                 (true_object_box_coordinate[..., 1] + true_object_box_coordinate[..., 3] / 2)], axis=-1)
        prediction_object_box_normalized_coordinate = tf.stack(values=[(prediction_object_box_coordinate[..., 0] - prediction_object_box_coordinate[..., 2] / 2),
                                                                       (prediction_object_box_coordinate[..., 1] - prediction_object_box_coordinate[..., 3] / 2),
                                                                       (prediction_object_box_coordinate[..., 0] + prediction_object_box_coordinate[..., 2] / 2),
                                                                       (prediction_object_box_coordinate[..., 1] + prediction_object_box_coordinate[..., 3] / 2), ], axis=-1)
        prediction_iou = IOU.iou(true_object_box_normalized_coordinate,  prediction_object_box_normalized_coordinate)
        # 分别得到网络输出结果中的置信度、类别以及坐标
        """++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"""
        # y_prediction = tf.reshape(tensor=y_prediction, shape=(-1, self.grid_size[0], self.grid_size[1], 5 * self.number_of_grid_box + self.numer_of_object_classes))
        # # prediction_object_confidence = y_prediction[:, :, :, :self.number_of_grid_box]
        #
        # prediction_object_confidence = tf.gather(y_prediction, [0, 5], axis=-1)
        # prediction_object_classes = y_prediction[:, :, :, 5 * self.number_of_grid_box:]
        # # prediction_object_box_coordinate = y_prediction[:, :, :, self.number_of_grid_box:5*self.number_of_grid_box]
        # prediction_object_box_coordinate = tf.gather(y_prediction, [1, 2, 3, 4, 6, 7, 8, 9], axis=-1)
        # # 分别得到真实标签输出结果中的置信度、类别以及坐标
        # true_object_confidence = y_true[:, :, :, :1]
        # true_object_box_coordinate = y_true[:, :, :, 1:5]
        # true_object_classes = y_true[:, :, :, 5:]
        # # print("====", type(true_object_box_coordinate))
        # # 分别重塑网络输出结果和真实标签的张量维度，其中真实标签中的坐标一项进行了扩充，重塑后的网络输出结果和真实标签维度一致
        # true_object_box_coordinate = tf.reshape(true_object_box_coordinate, shape=[-1, self.grid_size[0], self.grid_size[1], 1, 4])
        # true_object_box_coordinate = tf.tile(true_object_box_coordinate, multiples=[1, 1, 1, self.number_of_grid_box, 1])
        # prediction_object_box_coordinate = tf.reshape(prediction_object_box_coordinate, shape=[-1, self.grid_size[0], self.grid_size[1], self.number_of_grid_box, 4])
        # # 对每一个边框需要预测其（x, y, w, h），其中（x, y）是相对于负责预测的grid左上角坐标的偏移量，（w,h）是相对于整个图像的宽度和高度，因此需要构建预测值和真实值的中心点（x,y）的坐标相对于每个网格左上角的坐标系，并对（x,y）进行处理
        # # 偏移量与中心点公式为: offset_x_coordinate = (x_coordinate * cell_size) / image_size - offset
        # grid_x_range = tf.range(self.grid_size[0], dtype=tf.float32)
        # grid_y_range = tf.range(self.grid_size[1], dtype=tf.float32)
        # grid_x, grid_y = tf.meshgrid(grid_x_range, grid_y_range)
        # x_coordinate_offset = tf.reshape(grid_x, (-1, 1))
        # y_coordinate_offset = tf.reshape(grid_y, (-1, 1))
        # x_y_coordinate_offset = tf.concat([x_coordinate_offset, y_coordinate_offset], axis=-1)
        # x_y_coordinate_offset = tf.cast(tf.reshape(x_y_coordinate_offset, [self.grid_size[0], self.grid_size[1], 1, 2]), tf.float32)
        # prediction_box_normalized_coordinate = tf.stack(
        #     values=[(prediction_object_box_coordinate[:, :, :, :, 0] + x_y_coordinate_offset[..., 0]) / self.grid_size[0], (prediction_object_box_coordinate[:, :, :, :, 1] + x_y_coordinate_offset[..., 1]) / self.grid_size[0],
        #             tf.square(prediction_object_box_coordinate[:, :, :, :, 2]), tf.square(prediction_object_box_coordinate[:, :, :, :, 3])], axis=-1)
        # true_box_normalized_coordinate = tf.stack(values=[true_object_box_coordinate[:, :, :, :, 0] * self.grid_size[0] - x_y_coordinate_offset[..., 0], true_object_box_coordinate[:, :, :, :, 1] * self.grid_size[0] - x_y_coordinate_offset[..., 0],
        #                                                   tf.sqrt(true_object_box_coordinate[:, :, :, :, 2]), tf.sqrt(true_object_box_coordinate[:, :, :, :, 3])], axis=-1)
        # # 计算预测值和真实值之间的 IOU, 每个真实框和预测框都应计算出一个 IOU 值，因此最终的 IOU 格式应该为:[batch_size, 7, 7, 2]
        # prediction_box_normalized_coordinate = tf.stack(
        #     [prediction_box_normalized_coordinate[..., 0] - prediction_box_normalized_coordinate[..., 2] / 2.0, prediction_box_normalized_coordinate[..., 1] - prediction_box_normalized_coordinate[..., 3] / 2.0,
        #      prediction_box_normalized_coordinate[..., 0] + prediction_box_normalized_coordinate[..., 2] / 2.0, prediction_box_normalized_coordinate[..., 1] + prediction_box_normalized_coordinate[..., 3] / 2.0], axis=-1)
        # true_box_normalized_coordinate = tf.stack([true_box_normalized_coordinate[..., 0] - true_box_normalized_coordinate[..., 2] / 2.0, true_box_normalized_coordinate[..., 1] - true_box_normalized_coordinate[..., 3] / 2.0,
        #                                            true_box_normalized_coordinate[..., 0] + true_box_normalized_coordinate[..., 2] / 2.0, true_box_normalized_coordinate[..., 1] + true_box_normalized_coordinate[..., 3] / 2.0], axis=-1)
        # prediction_iou = IOU.iou(prediction_object_box_normalized_coordinate, true_object_box_normalized_coordinate)
        # prediction_iou = IOU.bbox_iou(true_box_normalized_coordinate, prediction_box_normalized_coordinate)
        # 通过iou构造标签， 有目标的那个cell的iou大的boxes负责预测，其余不预测
        predictor_mask = tf.reduce_max(prediction_iou, axis=3, keepdims=True)
        predictor_mask = tf.cast(prediction_iou >= predictor_mask, tf.float32) * true_object_confidence
        no_object_mask = tf.ones_like(predictor_mask) - predictor_mask

        # 计算类别损失
        class_loss = tf.reduce_mean(tf.reduce_sum(tf.square(true_object_confidence * (true_object_classes - prediction_object_classes)), axis=[1, 2, 3]))

        # 计算置信度损失
        obj_loss = tf.reduce_mean(tf.reduce_sum(tf.square(predictor_mask * (prediction_object_confidence - prediction_iou)), axis=[1, 2, 3]))
        no_object_loss = tf.reduce_mean(tf.reduce_sum(tf.square(no_object_mask * prediction_object_confidence), axis=[1, 2, 3]))

        # 计算位置坐标损失
        predictor_mask = predictor_mask[:, :, :, :, None]
        location_loss = tf.reduce_mean(tf.reduce_sum(tf.square(predictor_mask * (true_object_box_coordinate - prediction_object_box_coordinate)), axis=[1, 2, 3]))

        # 最终损失函数
        loss = self.lambda_coord * location_loss + self.lambda_coord * obj_loss + self.lambda_none_object * no_object_loss + class_loss
        return loss

    # def compute_yolo_loss1(self, y_true, y_prediction):
    #     """
    #     计算 yolo 的损失函数
    #     :param y_true: 真实值标签，真实标签格式为 [confidence, x, y, w, h, class]
    #     :param y_prediction: 预测值标签, 其是一个 7*7*（5+20）的一维张量
    #     :return: 损失函数的值
    #     """
    #     # 分别得到网络输出结果中的置信度、类别以及坐标
    #     y_prediction = tf.reshape(tensor=y_prediction, shape=(-1, self.grid_size[0], self.grid_size[1], 5 * self.number_of_grid_box + self.numer_of_object_classes))
    #     prediction_object_confidence = tf.gather(y_prediction, indices=[0, 5], axis=3)
    #     prediction_object_classes = y_prediction[:, :, :, 5 * self.number_of_grid_box:]
    #     prediction_object_box_coordinate = tf.gather(y_prediction, indices=[1, 2, 3, 4, 6, 7, 8, 9], axis=3)
    #     # 分别得到真实标签输出结果中的置信度、类别以及坐标
    #     true_object_confidence = y_true[:, :, :, :1]
    #     true_object_box_coordinate = y_true[:, :, :, 1:5]
    #     true_object_classes = y_true[:, :, :, 5:]
    #     # 分别重塑网络输出结果和真实标签的张量维度，其中真实标签中的坐标一项进行了扩充，重塑后的网络输出结果和真实标签维度一致,用于计算最终的损失函数
    #     true_object_box_coordinate = tf.reshape(true_object_box_coordinate, shape=[-1, self.grid_size[0], self.grid_size[1], 1, 4])
    #     true_object_box_coordinate = tf.tile(true_object_box_coordinate, multiples=[1, 1, 1, self.number_of_grid_box, 1])
    #     prediction_object_box_coordinate = tf.reshape(prediction_object_box_coordinate, shape=[-1, self.grid_size[0], self.grid_size[1], self.number_of_grid_box, 4])
    #     # 对每一个边框需要预测其（x, y, w, h），其中（x, y）是相对于负责预测的grid左上角坐标的偏移量，（w,h）是相对于整个图像的宽度和高度，因此需要构建预测值和真实值的中心点（x,y）的坐标相对于每个网格左上角的坐标系,同时将坐标转换为[x_min,y_min,x_max,y_max]的格式
    #     # 处理方式为对坐标进行处理，计算每个坐标在相对于网格左上角的坐标的值，范围为 0~1，计算公式为: coordinate = (coordinate * cell_size) / image_size - offset
    #     # 构建网格坐标，并计算出每个偏移量 offset
    #     grid_x_range = tf.range(self.grid_size[0], dtype=tf.float32)
    #     grid_y_range = tf.range(self.grid_size[1], dtype=tf.float32)
    #     grid_x, grid_y = tf.meshgrid(grid_x_range, grid_y_range)
    #     x_coordinate_offset = tf.reshape(grid_x, (-1, 1))
    #     y_coordinate_offset = tf.reshape(grid_y, (-1, 1))
    #     x_y_coordinate_offset = tf.concat([x_coordinate_offset, y_coordinate_offset], axis=-1)
    #     x_y_coordinate_offset = tf.cast(tf.reshape(x_y_coordinate_offset, [self.grid_size[0], self.grid_size[1], 1, 2]), tf.float32)
    #     true_object_box_coordinate = tf.stack(values=[(true_object_box_coordinate[..., 0] * self.grid_size[0] / self.image_shape[0] - x_y_coordinate_offset[..., 0]),
    #                                                   (true_object_box_coordinate[..., 1] * self.grid_size[0] / self.image_shape[0] - x_y_coordinate_offset[..., 1]),
    #                                                   (tf.square(true_object_box_coordinate[..., 2] / self.image_shape[0])),
    #                                                   (tf.square(true_object_box_coordinate[..., 3] / self.image_shape[1]))], axis=-1)
    #     prediction_object_box_coordinate = tf.stack(values=[(prediction_object_box_coordinate[..., 0] * self.grid_size[0] / self.image_shape[0] - x_y_coordinate_offset[..., 0]),
    #                                                         (true_object_box_coordinate[..., 1] * self.grid_size[0] / self.image_shape[0] - x_y_coordinate_offset[..., 1]),
    #                                                         (tf.square(prediction_object_box_coordinate[..., 2] / self.image_shape[0])),
    #                                                         (tf.square(prediction_object_box_coordinate[..., 3] / self.image_shape[0]))], axis=-1)
    #     # 将坐标变换为[x_min,y_min,x_max,y_max]的格式，方便计算真实框和预测框之间的IOU
    #     true_object_box_normalized_coordinate = tf.stack(values=[(true_object_box_coordinate[..., 0] - true_object_box_coordinate[..., 2] / 2),
    #                                                              (true_object_box_coordinate[..., 1] - true_object_box_coordinate[..., 3] / 2),
    #                                                              (true_object_box_coordinate[..., 0] + true_object_box_coordinate[..., 2] / 2),
    #                                                              (true_object_box_coordinate[..., 1] + true_object_box_coordinate[..., 3] / 2)], axis=-1)
    #     prediction_object_box_normalized_coordinate = tf.stack(values=[(prediction_object_box_coordinate[..., 0] - prediction_object_box_coordinate[..., 2] / 2),
    #                                                                    (prediction_object_box_coordinate[..., 1] - prediction_object_box_coordinate[..., 3] / 2),
    #                                                                    (prediction_object_box_coordinate[..., 0] + prediction_object_box_coordinate[..., 2] / 2),
    #                                                                    (prediction_object_box_coordinate[..., 1] + prediction_object_box_coordinate[..., 3] / 2), ], axis=-1)
    #     # # 计算预测框和真实框的IOU值，确定由那个一个框来负责预测目标
    #     iou_calculate_mask = IOU1.iou(true_object_box_normalized_coordinate, prediction_object_box_normalized_coordinate)
    #
    #     response_for_object_prediction_box_index = tf.expand_dims(tf.argmax(iou_calculate_mask, axis=-1), axis=-1)
    #     response_for_none_object_prediction_box_index = tf.expand_dims(tf.argmin(iou_calculate_mask, axis=-1), axis=-1)
    #     # 计算类别损失
    #     class_loss = losses.mean_squared_error(true_object_classes, prediction_object_classes)
    #     # 计算置信度损失
    #     confidence_loss = (self.lambda_coord * losses.mean_squared_error(true_object_confidence[:, :, :, 0], prediction_object_confidence[:, :, :, 0]) +
    #                        self.lambda_none_object * losses.mean_squared_error(true_object_confidence[:, :, :, 0], prediction_object_confidence[:, :, :, 0]))
    #
    #     # 计算位置坐标损失
    #     location_loss = self.lambda_coord * (losses.mean_squared_error(true_object_box_normalized_coordinate[:, :, :, :, 0][0], prediction_object_box_normalized_coordinate[:, :, :, :, 0][0]) +
    #                                          losses.mean_squared_error(true_object_box_normalized_coordinate[:, :, :, :, 0][1], prediction_object_box_normalized_coordinate[:, :, :, :, 0][1]) +
    #                                          losses.mean_squared_error(true_object_box_normalized_coordinate[:, :, :, :, 0][2], prediction_object_box_normalized_coordinate[:, :, :, :, 0][2]) +
    #                                          losses.mean_squared_error(true_object_box_normalized_coordinate[:, :, :, :, 0][3], prediction_object_box_normalized_coordinate[:, :, :, :, 0][3]))
    #
    #     # 最终损失函数
    #     loss = location_loss + confidence_loss + class_loss
    #     return loss
