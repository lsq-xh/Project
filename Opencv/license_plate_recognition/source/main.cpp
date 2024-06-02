#include<iostream>
#include<string>
#include<vector>
#include<opencv2/opencv.hpp>
#include"../detection.h"


using namespace std;
using namespace cv;

int main()
{
	cout << "begain to detection" << endl;
	LicensePlateNumberDetection exe;
	string detectionImagePath = "E:/CV/license_plate_recognition/resources/car.jpg";
	Mat detectionImage  = exe.readDetectionImage(detectionImagePath);
	vector<Rect>licesePlateLocationResult = exe.licensePlateLocate(detectionImage);
	vector<Mat>licensePlateSegmentResult = exe.licensePlateSegment(licesePlateLocationResult, detectionImage);
	//string  detectionResult = exe.licensePlateRecongnize(licensePlateSegmentResult);
	//cout << "³µÅÆºÅÎª£º" << detectionResult << endl;
	system("pause");
	cout << "stop detection" << endl;
	return 0;
}