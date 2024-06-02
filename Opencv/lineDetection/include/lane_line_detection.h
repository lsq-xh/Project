#ifndef _LANELINEDETECTION_H_
#define _LANELINEDETECTION_H_

#include<iostream>
#include<string>
#include<tuple>
#include<vector>
#include<opencv2/opencv.hpp>


class laneLineDetection
{
	public:
		laneLineDetection();
		void plotLaneLineDetectionResult(cv::Mat toPlotImage, std::vector<cv::Point>laneLinePoint);
		std::pair<std::vector<cv::Vec4f>, std::vector<cv::Vec4f>> laneLineDetectionFunction(cv::Mat toDetectoinImage, float filterThresh);
		std::vector<cv::Point> laneLineCurveFitting(std::vector<cv::Point> laneLinePoint, int fittingOrder, cv::Mat toDetectionImage);
		std::vector<cv::Point> analysisLaneLinePoint(std::vector<cv::Vec4f> laneLinePoint);
		~laneLineDetection();
	private:
	protected:
};



#endif // !LANELINEDETECTION

