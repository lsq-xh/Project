#include<iostream>
#include<vector>
#include<utility>
#include<numeric>
#include<opencv2/opencv.hpp>
#include"../include/lane_line_detection.h"

using namespace cv;
using namespace std;

laneLineDetection::laneLineDetection()
{

}
/*
illustrate:
	对要检测车道的图像进行处理，得到车道线的线段坐标并返回
param:
	totoDetectoinImage: 要检测的图像
	filterThresh：对车道线进行离群值剔除的阈值
return:
	要进行拟合的左右车道线线段端点坐标
*/
pair<vector<Vec4f>, vector<Vec4f>> laneLineDetection::laneLineDetectionFunction(Mat toDetectoinImage, float filterThresh)
{
	Mat todetectionImageEdges;
	GaussianBlur(toDetectoinImage, toDetectoinImage,Size(3, 3),0,0);
	Canny(toDetectoinImage, todetectionImageEdges, 240, 140);
	Mat maskImage = Mat::zeros(cv::Size(toDetectoinImage.cols, toDetectoinImage.rows),CV_8UC1);
	/* 确定要保留和填充的区域，即排除图片中的大部分非车道区域，保留图片中的车道区域 */
	Point maskVertex[4];
	maskVertex[3] = Point(0, toDetectoinImage.rows);
	maskVertex[0] = Point(int(0.15 * toDetectoinImage.cols),0.5*toDetectoinImage.rows);
	maskVertex[2] = Point(toDetectoinImage.cols, toDetectoinImage.rows);
	maskVertex[1] = Point(int(0.85 * toDetectoinImage.cols),0.5 * toDetectoinImage.rows);
	vector<Point> maskVertexPoint;
	for (int maskVertexIndex = 0; maskVertexIndex < 4; maskVertexIndex++)
	{
		maskVertexPoint.push_back(maskVertex[maskVertexIndex]);
	}
	Scalar lineColor = (255, 255, 255);

	/* 根据设定的掩膜对原图像进行填充，并将原图像和掩膜图像进行运算，获得二值图像 */
	fillPoly(maskImage, maskVertexPoint, lineColor);
	Mat  maskedToDetectionImage;
	bitwise_and(todetectionImageEdges, maskImage, maskedToDetectionImage);

	/* 对图像进行形态学变换、滤波等操作、数值运算等操作，消除过多的杂乱线条 */
	Mat morphologyMaskedToDetectionImage;
	Mat kernelDilate = getStructuringElement(MORPH_RECT, Size(5, 5));
	morphologyEx(maskedToDetectionImage, morphologyMaskedToDetectionImage, MORPH_DILATE, kernelDilate);
	Mat kernelErode = getStructuringElement(MORPH_RECT, Size(5, 5));
	morphologyEx(morphologyMaskedToDetectionImage, morphologyMaskedToDetectionImage, MORPH_ERODE, kernelErode);
	Mat kernelErodeAgain = getStructuringElement(MORPH_RECT, Size(5, 5));
	Mat morphologyMaskedToDetectionImageLast;
	morphologyEx(todetectionImageEdges, morphologyMaskedToDetectionImageLast, MORPH_DILATE, kernelErodeAgain);

	//subtract(morphologyMaskedToDetectionImage, morphologyMaskedToDetectionImageLast, morphologyMaskedToDetectionImage);
	/* 进一步处理，消除干扰因素 */
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(morphologyMaskedToDetectionImage, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	for(int contoursIndex = 0; contoursIndex< contours.size(); contoursIndex++)
	{
		RotatedRect rect = minAreaRect(contours[contoursIndex]);
		Rect boundrect = boundingRect(contours[contoursIndex]);
		float width = rect.size.width;
		float height = rect.size.height;
		if (height > width)
		{
			float temp = width;
			width = height;
			height = temp;
		}
		if (height / width > 0.1 )
		{
			morphologyMaskedToDetectionImage(boundrect).setTo(0);
		}
	}
	Mat kernel1 = getStructuringElement(MORPH_RECT, Size(3, 3));
	morphologyEx(morphologyMaskedToDetectionImage, morphologyMaskedToDetectionImage, MORPH_ERODE, kernel1);
	Mat kernel2 = getStructuringElement(MORPH_RECT, Size(5, 5));
	morphologyEx(morphologyMaskedToDetectionImage, morphologyMaskedToDetectionImage, MORPH_DILATE, kernel2);

	/* 进行霍夫变换，并求取每一条线段的斜率和每一条线段端点的坐标，分左右车道线分别进行 */
	vector <Vec4f> lineDetectedInImage;
	HoughLinesP(morphologyMaskedToDetectionImage, lineDetectedInImage, 1, CV_PI / 180, 50, 0, 0);
	vector<Vec4f> leftLaneLine;
	vector<Vec4f> rightLaneLine;
	vector<float> leftLaneLineSlpope;
	vector<float> rightLaneLineSlope;
	for (size_t laneLineIndex = 0; laneLineIndex < lineDetectedInImage.size(); laneLineIndex++)
	{
		line(toDetectoinImage,Point(lineDetectedInImage[laneLineIndex][0], lineDetectedInImage[laneLineIndex][1]), Point(lineDetectedInImage[laneLineIndex][2],lineDetectedInImage[laneLineIndex][3]), Scalar(0, 0, 255), 2, 2);
		Vec4f toCalculateLine = lineDetectedInImage[laneLineIndex];
		float laneLineSlope = (toCalculateLine[3] - toCalculateLine[1]) / (toCalculateLine[2] - toCalculateLine[0]);
		if (laneLineSlope < 0)
		{
			rightLaneLine.push_back(toCalculateLine);
			rightLaneLineSlope.push_back(laneLineSlope);
		}
		else if(laneLineSlope > 0)
		{
			leftLaneLine.push_back(toCalculateLine);
			leftLaneLineSlpope.push_back(laneLineSlope);
		}
		else
		{
			/* 不进行任何操作 */
			//lineDetectedInImage.erase(lineDetectedInImage.begin()+ laneLineIndex);

		}
	}

	/* 计算霍夫变换拟合出的每一条线的斜率，剔除明显的离群值，保留斜率基本相同的线段用作后续的拟合计算，分左右车道线分别进行剔除 */
	float leftLaneLineSlpopeAverage = accumulate(leftLaneLineSlpope.begin(), leftLaneLineSlpope.end(), 0.0)/leftLaneLineSlpope.size();	
	float rightLaneLineSlpopeAverage = accumulate(rightLaneLineSlope.begin(), rightLaneLineSlope.end(), 0.0)/rightLaneLineSlope.size();
	for (int leftLaneLineIndex = 0; leftLaneLineIndex < leftLaneLine.size(); leftLaneLineIndex++)
	{
		float diff = leftLaneLineSlpope[leftLaneLineIndex] - leftLaneLineSlpopeAverage;
		if ( abs(diff) > filterThresh)
		{
			leftLaneLineSlpope.erase(leftLaneLineSlpope.begin()+leftLaneLineIndex);
			leftLaneLine.erase(leftLaneLine.begin() + leftLaneLineIndex);
		}
	}
	for (int rightLaneLineIndex = 0; rightLaneLineIndex < rightLaneLine.size(); rightLaneLineIndex++)
	{
		float diff = rightLaneLineSlope[rightLaneLineIndex] - rightLaneLineSlpopeAverage;
		if (abs(diff) > filterThresh)
		{
			rightLaneLineSlope.erase(rightLaneLineSlope.begin() + rightLaneLineIndex);
			rightLaneLine.erase(rightLaneLine.begin() + rightLaneLineIndex);
		}
	}
	return std::make_pair(rightLaneLine, leftLaneLine);
}

/*
illustrate:
	将车道线坐标解析成每一个点的形式，并存储在vector向量中
param:
	laneLinePoint: 要解析的车道线坐标
return:
	解析后的车道线坐标
*/
vector<Point> laneLineDetection::analysisLaneLinePoint(vector<Vec4f> laneLinePoint)
{
	vector<Point>analysisedLinePoint;
	/* 读取已有车道线中的每一个车道线的两端端点，并将所有点坐标存放在返回值中 */
	for (const auto& element : laneLinePoint)
	{
		analysisedLinePoint.push_back(Point(element[0], element[1]));
		analysisedLinePoint.push_back(Point(element[2], element[3]));
	}

	return analysisedLinePoint;
}

/*
illustrte:
	根据检测到的车道线坐标，利用最小二乘法进行拟合，得到最终的车道线
param:
	lanelinePoint: 图像中检测到的左右车道线的坐标
	fittingOrder: 车道线拟合的阶数
	fittingResult：车道线拟合的结果的系数矩阵
return:
	车道线拟合的结果的系数矩阵
*/
vector<Point> laneLineDetection::laneLineCurveFitting(vector<Point> laneLinePoint, int fittingOrder, Mat toDetectionImage)
{
	/* 构造opencv线性方程求解函数输入矩阵X */
	int pointNumber = laneLinePoint.size();
	cv::Mat inputArrayX = Mat::zeros(fittingOrder + 1, fittingOrder + 1, CV_64F);
	for (int i = 0; i < fittingOrder + 1; i++)
	{
		for (int j = 0; j < fittingOrder + 1; j++)
		{
			for (int k = 0; k < pointNumber; k++)
			{
				inputArrayX.at<double>(i, j) = inputArrayX.at<double>(i, j) + std::pow(laneLinePoint[k].x, i + j);
			}
		}
	}

	/*构造opencv线性方程求解函数输入矩阵Y */
	cv::Mat inputArrayY = cv::Mat::zeros(fittingOrder + 1, 1, CV_64F);
	for (int i = 0; i < fittingOrder + 1; i++)
	{
		for (int k = 0; k < pointNumber; k++)
		{
			inputArrayY.at<double>(i, 0) = inputArrayY.at<double>(i, 0) +std::pow(laneLinePoint[k].x, i) * laneLinePoint[k].y;
		}
	}
	/* 求解曲线拟合系数矩阵 */
	Mat fittingResultMatrix = Mat::zeros(fittingOrder + 1, 1, CV_64FC1);
	cv::solve(inputArrayX, inputArrayY, fittingResultMatrix, cv::DECOMP_LU);

	/* 根据系数矩阵，求解拟合后的曲线坐标 */
	std::vector<cv::Point> points_fitted;
	for (Point element : laneLinePoint)
	{
		double pointY = 0.;
		double pointX = element.x;
		for (int orderIndex=0; orderIndex <fittingOrder; orderIndex++)
		{
			pointY = pointY + fittingResultMatrix.at<double>(orderIndex, 0) * std::pow(pointX, orderIndex);
		}
		//points_fitted.push_back(cv::Point(pointX, pointY));
	}

	return points_fitted;
}


/*
illustrate:
	根据拟合得到的车道线坐标点，在原图上绘制显示最终结果
param:
	toPlotImage: 要绘制车道线的图像
	laneLinePoint：车道线检测的最终结果
return:
	不返回任何值
*/
void laneLineDetection::plotLaneLineDetectionResult(Mat toPlotImage, vector<Point>laneLinePoint)
{
	polylines(toPlotImage, laneLinePoint, false, cv::Scalar(0, 255, 255), 8, 8, 0);
}

laneLineDetection::~laneLineDetection()
{

}