from utils import calibration_image_gather
from monocular_camera_calibration import MonocularCameraCalibration
from binocular_camera_calibration import BinocularCameraCalibration


def which_method_to_choose(mode, calibration_image_path, camera_parameter_save_dir, camera_index):
    """
    mode=0进行单目相机标定，mode=1进行双目相机标定
    :param mode: 选择进行双目相机标定或单目相机标定，0则是双目相机标定，1则是单目相机标定
    :param camera_index: 相机索引
    :param calibration_image_path: 标定图像路径
    :param camera_parameter_save_dir: 标定结果保存路径
    :return: None
    """
    if mode == 0:
        left_camera_calibration_image = calibration_image_gather(mode=mode, camera_index=camera_index, image_save_path=calibration_image_path)[0]
        right_camera_calibration_image = calibration_image_gather(mode=mode, camera_index=camera_index, image_save_path=calibration_image_path)[1]
        camera_calibrate = BinocularCameraCalibration(camera_parameter_save_dir=camera_parameter_save_dir, corner_number_each_row=6, corner_number_each_column=4, right_camera_calibration_image_path=right_camera_calibration_image,
                                                      left_camera_calibration_image_path=left_camera_calibration_image)
        camera_calibrate.binocular_camera_calibration()
    elif mode == 1:
        camera_calibrate = MonocularCameraCalibration(calibration_image_path=calibration_image_path, camera_parameter_save_dir=camera_parameter_save_dir, corner_number_each_row=6, corner_number_each_column=4)
        camera_calibrate.zhang_zheng_you_camera_calibration_method()
    else:
        print("请重新数据，单目相机标定请输入0，双目相机标定请输入1")


def main():
    calibration_image_path = r""
    camera_parameter_save_dir = r""
    which_method_to_choose(mode=0, calibration_image_path=calibration_image_path, camera_parameter_save_dir=camera_parameter_save_dir, camera_index=1)


if __name__ == "__main__":
    main()

