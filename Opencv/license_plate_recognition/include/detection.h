#ifndef  DETECTION_H
#define  DETECTION_H


#include<iostream>
#include<string>
#include<opencv2/opencv.hpp>


class LicensePlateNumberDetection
{
	public:
		LicensePlateNumberDetection();
		~LicensePlateNumberDetection();
		cv::Mat readDetectionImage(std::string imagePath);
		std::vector<cv::Rect> licensePlateLocate(cv::Mat image);
		std::vector<cv::Mat> licensePlateSegment(std::vector<cv::Rect>, cv::Mat toDetectionImage);
		std::string licensePlateRecongnize(std::vector<cv::Mat>toRecongnizeImage);
		
	private:
	protected:
};
#endif // ! DETECTION_H
