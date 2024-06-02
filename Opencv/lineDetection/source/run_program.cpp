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
	�ж��Ƿ�������ͷ�ɵ��ã��Կɵ�������ͷID���б�Ų��������IDֵ
	��������ͷ�ɵ������ȡ������Ƶ�ļ�
param:
	cameraNumber: Ҫ���õ�����ͷ
return:
	ʵ��Ҫ���õ�����ͷ��������ʹ�ñ�����Ƶ�ļ��ı�ʶ
*/
int runProgram::usingCameraOrVideo(int cameraNumber)
{
	vector<int> cameraListIndex;
	VideoCapture camera;
	/*���ҿɵ�������ͷ����������ͷID���б��*/
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
	/*�ж���ʹ������ͷ���Ǳ����ļ�*/
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
	��ȡ��Ƶ�ļ�·��
return:
	������Ƶ�ļ��Ĵ洢·��
*/
string runProgram::getVideoFilePath()
{
	return videoFilePath;
}

/*
illustrate:
	����Ƶ֡������ʾ
return:
	�������κ�ֵ
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