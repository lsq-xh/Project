import os
import cv2 as cv
import numpy as np

global left_camera_gray
global right_camera_gray


class BinocularCameraCalibration:
    def __init__(self, camera_parameter_save_dir, corner_number_each_row, corner_number_each_column, right_camera_calibration_image_path, left_camera_calibration_image_path):
        """
        初始话函数
        :param  right_camera_calibration_image_path: 右侧相机拍摄的标定板图像
        :param  right_camera_calibration_image_path: 左侧相机拍摄的标定板图像
        :param camera_parameter_save_dir: 被标定相机内参和外参保存的路径
        :param corner_number_each_row: 标定板每一行的角点数,根据标定板确定
        :param corner_number_each_column: 标定板每一列的角点数,根据标定板确定
        """
        self.camera_parameter_save_dir = camera_parameter_save_dir
        self.corner_number_each_row = corner_number_each_row
        self.corner_number_each_column = corner_number_each_column
        self.right_camera_calibration_image_path = right_camera_calibration_image_path
        self.left_camera_calibration_image_path = left_camera_calibration_image_path

    def get_object_point(self):
        global left_camera_gray
        global right_camera_gray
        # corner_number_each_row 和corner_number_each_column是自定义的。例程用的棋盘足够大包含了7×6以上个角点，这里用的只有6×4。这里如果角点维数超出的话，标定的时候会报错。
        # 准备目标点坐标，其格式为： (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros(shape=(self.corner_number_each_row * self.corner_number_each_column, 3), dtype=np.float32)
        # 设定世界坐标下点的坐标值，因为用的是棋盘可以直接按网格取；假定棋盘正好在x-y平面上，这样z值直接取0，简化初始化步骤。
        # 把列向量[0:corner_number_each_row]复制了corner_number_each_column列，把行向量[0:corner_number_each_column]复制了corner_number_each_row行。
        # 转置reshape后，每行都是4×6网格中的某个点的坐标。
        objp[:, :2] = np.mgrid[0:self.corner_number_each_row, 0:self.corner_number_each_column].T.reshape(-1, 2)
        # 3d point in real world space
        left_camera_object_points = []
        right_camera_object_points = []
        # 2d points in image plane.
        left_camera_image_points = []
        right_camera_image_points = []
        left_images = os.listdir(self.left_camera_calibration_image_path)
        right_images = os.listdir(self.right_camera_calibration_image_path)
        for file_name in left_images:
            # 对每张图片，识别出角点，记录世界物体坐标和图像坐标
            left_camera_image = cv.imread(self.left_camera_calibration_image_path + file_name)
            left_camera_gray = cv.cvtColor(left_camera_image, cv.COLOR_BGR2GRAY)
            # 寻找角点，存入corners，ret是找到角点的flag
            ret, corners = cv.findChessboardCorners(left_camera_gray, (self.corner_number_each_row, self.corner_number_each_column), None)
            if ret:
                left_camera_object_points.append(objp)
                # criteria:角点精准化迭代过程的终止条件
                criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                # 执行亚像素级角点检测
                corners2_left_camera = cv.cornerSubPix(left_camera_gray, corners, (11, 11), (-1, -1), criteria)
                left_camera_image_points.append(corners2_left_camera)

        for file_name in right_images:
            # 对每张图片，识别出角点，记录世界物体坐标和图像坐标
            right_camera_image = cv.imread(self.right_camera_calibration_image_path + file_name)
            right_camera_gray = cv.cvtColor(right_camera_image, cv.COLOR_BGR2GRAY)
            # 寻找角点，存入corners，ret是找到角点的flag
            ret, corners = cv.findChessboardCorners(right_camera_gray, (6, 4), None)
            if ret:
                right_camera_object_points.append(objp)
                # criteria:角点精准化迭代过程的终止条件
                criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                # 执行亚像素级角点检测
                corners2_right_camera = cv.cornerSubPix(right_camera_gray, corners, (11, 11), (-1, -1), criteria)
                right_camera_image_points.append(corners2_right_camera)

        image_points_left = np.array(left_camera_image_points)
        image_points_right = np.array(right_camera_image_points)
        left_camera_object_points = np.array(left_camera_object_points)
        right_camera_object_points = np.array(right_camera_object_points)
        return image_points_left, image_points_right, left_camera_object_points, right_camera_object_points

    def zhang_zheng_you_camera_calibration_method(self):
        """
        采用张正友相机标定法对相机的内参和外参进行标定
        :return:相机内参、畸变系数、旋转矩阵、平移矩阵
        """
        left_camera_image_points = self.get_object_point()[0]
        right_camera_image_points = self.get_object_point()[1]
        left_camera_object_points = self.get_object_point()[2]
        right_camera_object_points = self.get_object_point()[3]
        # 传入所有图片各自角点的三维、二维坐标，相机标定。每张图片都有自己的旋转和平移矩阵，但是相机内参和畸变系数只有一组。
        # 相机内参:camera_intrinsic_parameters
        # 畸变系数:camera_distortion_coefficient
        # 旋转矩阵:rotation_matrix
        # 平移矩阵:translation_matrix
        ret_left, left_camera_intrinsic_parameters, left_camera_distortion_coefficient, left_rotation_matrix, left_translation_matrix = cv.calibrateCamera(left_camera_object_points, left_camera_image_points, left_camera_gray.shape[::-1], None, None)
        ret_right, right_camera_intrinsic_parameters, right_camera_distortion_coefficient, right_rotation_matrix, right_translation_matrix = cv.calibrateCamera(right_camera_object_points, right_camera_image_points, right_camera_gray.shape[::-1], None, None)
        # 打印要求的两个矩阵参数
        print("camera_intrinsic_parameters:\n", left_camera_intrinsic_parameters)
        print("camera_distortion_coefficient:\n", left_camera_distortion_coefficient)
        print("rotation_matrix\n", left_rotation_matrix)
        print("translation_matrix\n", left_translation_matrix)
        print("camera_intrinsic_parameters:\n", right_camera_intrinsic_parameters)
        print("camera_distortion_coefficient:\n", right_camera_distortion_coefficient)
        print("rotation_matrix\n", right_rotation_matrix)
        print("translation_matrix\n", right_translation_matrix)
        # 计算误差
        total_error = 0
        for i in range(len(left_camera_object_points)):
            left_image_points_2, _ = cv.projectPoints(left_camera_object_points[i], left_rotation_matrix[i], left_translation_matrix[i], left_camera_intrinsic_parameters, left_camera_distortion_coefficient)
            error = cv.norm(left_camera_image_points[i], left_image_points_2, cv.NORM_L2) / len(left_image_points_2)
            total_error += error

        for i in range(len(right_camera_object_points)):
            right_image_points_2, _ = cv.projectPoints(right_camera_object_points[i], right_rotation_matrix[i], right_translation_matrix[i], right_camera_intrinsic_parameters, right_camera_distortion_coefficient)
            error = cv.norm(right_camera_image_points[i], right_image_points_2, cv.NORM_L2) / len(right_image_points_2)
            total_error += error

        # 保存相机标定参数
        with open(self.camera_parameter_save_dir, 'w') as f_left_camera:
            f_left_camera.write("camera_intrinsic_parameters:" + '\n' + str(left_camera_intrinsic_parameters) + '\n')
            f_left_camera.write("camera_distortion_coefficient:" + '\n' + str(left_camera_distortion_coefficient) + '\n')
            f_left_camera.close()

        with open(self.camera_parameter_save_dir, 'w') as f_right_camera:
            f_right_camera.write("camera_intrinsic_parameters:" + '\n' + str(right_camera_intrinsic_parameters) + '\n')
            f_right_camera.write("camera_distortion_coefficient:" + '\n' + str(right_camera_distortion_coefficient) + '\n')
            f_right_camera.close()

        left_camera_parameter = [left_camera_intrinsic_parameters, left_camera_distortion_coefficient, left_rotation_matrix, left_translation_matrix, left_camera_image_points]
        right_camera_parameter = [right_camera_intrinsic_parameters, right_camera_distortion_coefficient, right_rotation_matrix, right_translation_matrix, right_camera_image_points]
        return left_camera_parameter, right_camera_parameter

    def binocular_camera_calibration(self):
        left_camera_parameter = self.zhang_zheng_you_camera_calibration_method()[0]
        right_camera_parameter = self.zhang_zheng_you_camera_calibration_method()[1]
        image_points_left = left_camera_parameter[4]
        image_points_right = right_camera_parameter[4]
        object_points = self.get_object_point()[2]
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        flags = cv.CALIB_FIX_INTRINSIC
        ret, intrinsic_left_camera, distortion_left_camera, intrinsic_right_camera, distortion_right_camera, r, t, e, f = cv.stereoCalibrate(object_points, image_points_left,image_points_right,
                                                                                                                                             left_camera_parameter[0],left_camera_parameter[1],
                                                                                                                                             right_camera_parameter[0],right_camera_parameter[1],
                                                                                                                                             left_camera_gray.shape,criteria=criteria,flags=flags)
        with open(self.camera_parameter_save_dir, 'w') as f_binocular_camera:
            f_binocular_camera.write("left_camera_intrinsic_parameters:" + '\n' + str(intrinsic_left_camera) + '\n')
            f_binocular_camera.write("left_camera_distortion_coefficient:" + '\n' + str(distortion_left_camera) + '\n')
            f_binocular_camera.write("right_camera_intrinsic_parameters:" + '\n' + str(intrinsic_right_camera) + '\n')
            f_binocular_camera.write("right_camera_distortion_coefficient:" + '\n' + str(distortion_right_camera) + '\n')
            f_binocular_camera.write("right_camera_rotation_matrix_be_relative_to_left_camera:" + '\n' + str(r) + '\n')
            f_binocular_camera.write("right_camera_translate_matrix_be_relative_to_left_camera:" + '\n' + str(t) + '\n')
            f_binocular_camera.write("eigen_matrix:" + '\n' + str(e) + '\n')
            f_binocular_camera.write("fundamental_matrix:" + '\n' + str(f) + '\n')
            f_binocular_camera.close()

