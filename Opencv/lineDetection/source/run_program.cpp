#include<iostream>
#include<vector>
#include<algorithm>
#include<opencv2/opencv.hpp>
#include"../run_program.h"

using namespace std;
using namespace cv;


runProgram::runProgram()
{

}
/*
illustrate:
	判断是否有摄像头可调用，对可调用摄像头ID进行编号并返回最大ID值
	若无摄像头可调用则读取本地视频文件
param:
	cameraNumber: 要调用的摄像头
return:
	实际要调用的摄像头的索引或使用本地视频文件的标识
*/
int runProgram::usingCameraOrVideo(int cameraNumber)
{
	vector<int> cameraListIndex;
	VideoCapture camera;
	/*查找可调用摄像头，并对摄像头ID进行编号*/
	for(int cameraCount = 0;cameraCount<1000000;cameraCount++)
	{
		camera.open(cameraCount);
		if(camera.isOpened())
		{
			cameraListIndex.push_back(cameraCount);
			camera.release();
		}
		else
		{
			break;
		}
	}
	/*判断是使用摄像头还是本地文件*/
	if (cameraListIndex.size()==0)
	{
		std::cout << "no camera found, please check, now use video" << std::endl;
		return -1;
	}
	else
	{
		cout << "found camera, now use the input camera id" << endl;
		int cameraMaxId = *max_element(cameraListIndex.begin(), cameraListIndex.end());
		if (cameraNumber <= cameraMaxId && cameraNumber >= 0)
		{
			return cameraNumber;
		}
		else if (cameraNumber > cameraMaxId)
		{
			return 1;
		}
		else
		{
			return -1;
		}
	}
}
/*
illustrate:
	获取视频文件路径
return:
	本地视频文件的存储路径
*/
string runProgram::getVideoFilePath()
{
	return videoFilePath;
}

/*
illustrate:
	对视频帧进行显示
return:
	不返回任何值
*/
void runProgram::showFrame(Mat detectionFrame)
{
	namedWindow("detection_reuslt", WINDOW_AUTOSIZE);
	imshow("detection_reuslt", detectionFrame);
	if (waitKey(1) == 27)
	{
		destroyWindow("detection_reuslt");
	}
}

runProgram::~runProgram()
{

}