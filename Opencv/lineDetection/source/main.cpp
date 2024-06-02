#include<iostream>
#include<string>
#include<opencv2/opencv.hpp>
#include"../run_program.h"
#include"../lane_line_detection.h"

using namespace std;
using namespace cv;

int main()
{
	cout << "begain to detection" << endl;
	int usingCameraId = -4;
	runProgram aplicationExe;
	laneLineDetection imageProcess;

	int deviceChooseResult = aplicationExe.usingCameraOrVideo(usingCameraId);
	cout << "deviceChooseResult"<<deviceChooseResult << endl;
	if (deviceChooseResult == -1)
	{	
		VideoCapture cap(aplicationExe.getVideoFilePath());
		Mat detectionFrame;
		while (true)
		{
			
			cap.read(detectionFrame);
			resize(detectionFrame, detectionFrame, Size(720, 480));
			pair<vector<Vec4f>, vector<Vec4f>> laneLinePoints = imageProcess.laneLineDetectionFunction(detectionFrame, 0.01);
			vector<Point> rightLaneLinePoint = imageProcess.analysisLaneLinePoint(laneLinePoints.first);
			vector<Point> leftLaneLinePoint = imageProcess.analysisLaneLinePoint(laneLinePoints.second);
			vector<Point> fittingRightLaneLinePoint = imageProcess.laneLineCurveFitting(rightLaneLinePoint,3,detectionFrame);
			imageProcess.plotLaneLineDetectionResult(detectionFrame, fittingRightLaneLinePoint);
			vector<Point> fittingLeftLaneLinePoint = imageProcess.laneLineCurveFitting(leftLaneLinePoint, 3, detectionFrame);
			imageProcess.plotLaneLineDetectionResult(detectionFrame, fittingLeftLaneLinePoint);;
			aplicationExe.showFrame(detectionFrame);
		}
	}
	else
	{
		VideoCapture cap(deviceChooseResult);
		Mat detectionFrame;
		while (true)
		{
			cap.read(detectionFrame);
			resize(detectionFrame, detectionFrame, Size(720, 480));
			pair<vector<Vec4f>, vector<Vec4f>> laneLinePoints = imageProcess.laneLineDetectionFunction(detectionFrame, 0.01);
			vector<Point> rightLaneLinePoint = imageProcess.analysisLaneLinePoint(laneLinePoints.first);
			vector<Point> leftLaneLinePoint = imageProcess.analysisLaneLinePoint(laneLinePoints.second);
			vector<Point> fittingRightLaneLinePoint = imageProcess.laneLineCurveFitting(rightLaneLinePoint, 2, detectionFrame);
			imageProcess.plotLaneLineDetectionResult(detectionFrame, fittingRightLaneLinePoint);
			vector<Point> fittingLeftLaneLinePoint = imageProcess.laneLineCurveFitting(leftLaneLinePoint, 2, detectionFrame);
			imageProcess.plotLaneLineDetectionResult(detectionFrame, fittingLeftLaneLinePoint);;
			aplicationExe.showFrame(detectionFrame);
		}
	}

	system("pause");
	cout << "stop detection" << endl;
	return 0;
}