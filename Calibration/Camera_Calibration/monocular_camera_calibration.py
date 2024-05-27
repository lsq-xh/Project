import os
import cv2 as cv
import numpy as np

global gray


class MonocularCameraCalibration:
    """
    标定相机的内参和外参并将参数保存于Camera_Calibration.txt文件中
    """

    def __init__(self, calibration_image_path, camera_parameter_save_dir, corner_number_each_row, corner_number_each_column):
        """
        初始话函数
        :param calibration_image_path: 要标定的相机拍摄的标定板图像
        :param camera_parameter_save_dir: 被标定相机内参和外参保存的路径
        :param corner_number_each_row: 标定板每一行的角点数,根据标定板确定
        :param corner_number_each_column: 标定板每一列的角点数,根据标定板确定
        """
        self.calibration_image_path = calibration_image_path
        self.camera_parameter_save_dir = camera_parameter_save_dir
        self.corner_number_each_row = corner_number_each_row
        self.corner_number_each_column = corner_number_each_column

    def get_object_point(self):
        global gray
        # corner_number_each_row 和corner_number_each_column是自定义的。例程用的棋盘足够大包含了7×6以上个角点，这里用的只有6×4。这里如果角点维数超出的话，标定的时候会报错。
        # 准备目标点坐标，其格式为： (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros(shape=(self.corner_number_each_row * self.corner_number_each_column, 3), dtype=np.float32)
        # 设定世界坐标下点的坐标值，因为用的是棋盘可以直接按网格取；假定棋盘正好在x-y平面上，这样z值直接取0，简化初始化步骤。
        # 把列向量[0:corner_number_each_row]复制了corner_number_each_column列，把行向量[0:corner_number_each_column]复制了corner_number_each_row行。
        # 转置reshape后，每行都是4×6网格中的某个点的坐标。
        objp[:, :2] = np.mgrid[0:self.corner_number_each_row, 0:self.corner_number_each_column].T.reshape(-1, 2)
        # 3d point in real world space
        object_points = []
        # 2d points in image plane.
        image_points = []
        images = os.listdir(self.calibration_image_path)
        for file_name in images:
            # 对每张图片，识别出角点，记录世界物体坐标和图像坐标
            image = cv.imread(self.calibration_image_path + file_name)
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            # 寻找角点，存入corners，ret是找到角点的flag
            ret, corners = cv.findChessboardCorners(gray, (self.corner_number_each_row, self.corner_number_each_column), None)
            if ret:
                # criteria:角点精准化迭代过程的终止条件
                criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                # 执行亚像素级角点检测
                corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                object_points.append(objp)
                image_points.append(corners2)
        image_points = np.array(image_points)
        object_points = np.array(object_points)

        return image_points, object_points

    def zhang_zheng_you_camera_calibration_method(self):
        """
        采用张正友相机标定法对相机的内参和外参进行标定
        :return:相机内参、畸变系数、旋转矩阵、平移矩阵
        """
        image_points = self.get_object_point()[0]
        object_points = self.get_object_point()[1]
        # 传入所有图片各自角点的三维、二维坐标，相机标定。每张图片都有自己的旋转和平移矩阵，但是相机内参和畸变系数只有一组。
        # 相机内参:camera_intrinsic_parameters
        # 畸变系数:camera_distortion_coefficient
        # 旋转矩阵:rotation_matrix
        # 平移矩阵:translation_matrix
        ret, camera_intrinsic_parameters, camera_distortion_coefficient, rotation_matrix, translation_matrix = cv.calibrateCamera(object_points, image_points, gray.shape[::-1], None, None)
        # 打印要求的两个矩阵参数
        print("camera_intrinsic_parameters:\n", camera_intrinsic_parameters)
        print("camera_distortion_coefficient:\n", camera_distortion_coefficient)
        print("rotation_matrix\n", rotation_matrix)
        print("translation_matrix\n", translation_matrix)
        # 计算误差
        total_error = 0
        for i in range(len(object_points)):
            image_points_2, _ = cv.projectPoints(object_points[i], rotation_matrix[i], translation_matrix[i], camera_intrinsic_parameters, camera_distortion_coefficient)
            error = cv.norm(image_points[i], image_points_2, cv.NORM_L2) / len(image_points_2)
            total_error += error
        # 保存相机标定参数
        with open(self.camera_parameter_save_dir, 'w') as f:
            f.write("camera_intrinsic_parameters:" + '\n' + str(camera_intrinsic_parameters) + '\n')
            f.write("camera_distortion_coefficient:" + '\n' + str(camera_distortion_coefficient) + '\n')
            f.close()

        to_return_parameters = [camera_intrinsic_parameters, camera_distortion_coefficient, rotation_matrix, translation_matrix, image_points, object_points]
        return to_return_parameters
