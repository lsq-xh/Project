"""
调用摄像头采集照片
"""
import os
import cv2 as cv


def calibration_image_gather(mode, camera_index, image_save_path):
    """
    调用摄像头采集标定板图像
    :param mode: 选择进行单目相机标定还是双目相机标定，1--双目相机；0--单目相机
    :param camera_index: 要标定的摄像头的索引
    :param image_save_path: 标定图像保存的跟路径
    :return: None
    """
    left_camera_image_save_dir = image_save_path + "/" + "leftCamera/"
    right_camera_image_save_dir = image_save_path + "/" + "rightCamera/"
    if not os.path.exists(left_camera_image_save_dir):
        os.mkdir(left_camera_image_save_dir)
        os.mkdir(right_camera_image_save_dir)
    if len(os.listdir(left_camera_image_save_dir)) != 0 and len(os.listdir(right_camera_image_save_dir)) != 0:
        return left_camera_image_save_dir, right_camera_image_save_dir
    else:
        cap = cv.VideoCapture(camera_index)
        image_number_index = 0
        frame_width = 1280
        frame_height = 480
        if mode == 0:
            cap.set(cv.CAP_PROP_FRAME_WIDTH, frame_width)
            cap.set(cv.CAP_PROP_FRAME_HEIGHT, frame_height)
        elif mode == 1:
            cap.set(cv.CAP_PROP_FRAME_WIDTH, frame_width * 2)
            cap.set(cv.CAP_PROP_FRAME_HEIGHT, frame_height)
        else:
            print("do not need set the image width and height")
        while cap.isOpened():
            ret, image = cap.read()
            key = cv.waitKey(1) & 0xFF
            if mode == 0:
                if key == ord('s'):
                    cv.imwrite(image_save_path + str(image_number_index) + '.jpg', image)
                    print("图像已经保存")
                if key == ord('q'):
                    break
            elif mode == 1:
                image_capture_by_left_camera = image[:, 0:frame_width]
                image_capture_by_right_camera = image[:, frame_width:]
                if key == ord('s'):
                    print(key)
                    print(image_save_path)
                    print(left_camera_image_save_dir)
                    print(right_camera_image_save_dir)
                    if not os.path.exists(left_camera_image_save_dir):
                        os.makedirs(left_camera_image_save_dir)
                    if not os.path.exists(right_camera_image_save_dir):
                        os.makedirs(right_camera_image_save_dir)
                    cv.imwrite(left_camera_image_save_dir + str(image_number_index) + "_left.jpg", image_capture_by_left_camera)
                    cv.imwrite(right_camera_image_save_dir + str(image_number_index) + "_right .jpg", image_capture_by_right_camera)
                    print("图像已经保存")
                if key == ord('q'):
                    break
            else:
                print("please choose the right mode, it can be 1 or 0")
            image_number_index += 1
            cv.namedWindow("image captured by camera")
            cv.imshow("image captured by camera", image)
        cv.destroyAllWindows()
        print("done")
        return left_camera_image_save_dir, right_camera_image_save_dir
