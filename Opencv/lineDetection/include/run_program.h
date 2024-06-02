#ifndef RUNPROGRAM_H
#define RUNPROGRAM_H


#include<iostream>
#include<opencv2/opencv.hpp>

class runProgram
{
	public:
		runProgram();
		int usingCameraOrVideo(int cameraNumber);
		std::string getVideoFilePath();
		void showFrame(cv::Mat detectionFrame);
		~runProgram();
	private:
		std::string videoFilePath = "E:\\CV\\lane_detection\\resources\\SVID.mp4";
	protected:

};
#endif // !RUNPROGRAM_H

