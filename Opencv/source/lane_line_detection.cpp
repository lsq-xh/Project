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
	��Ҫ��⳵����ͼ����д����õ������ߵ��߶����겢����
param:
	totoDetectoinImage: Ҫ����ͼ��
	filterThresh���Գ����߽�����Ⱥֵ�޳�����ֵ
return:
	Ҫ������ϵ����ҳ������߶ζ˵�����
*/
pair<vector<Vec4f>, vector<Vec4f>> laneLineDetection::laneLineDetectionFunction(Mat toDetectoinImage, float filterThresh)
{
	Mat todetectionImageEdges;
	GaussianBlur(toDetectoinImage, toDetectoinImage,Size(3, 3),0,0);
	Canny(toDetectoinImage, todetectionImageEdges, 240, 140);
	Mat maskImage = Mat::zeros(cv::Size(toDetectoinImage.cols, toDetectoinImage.rows),CV_8UC1);
	/* ȷ��Ҫ�������������򣬼��ų�ͼƬ�еĴ󲿷ַǳ������򣬱���ͼƬ�еĳ������� */
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

	/* �����趨����Ĥ��ԭͼ�������䣬����ԭͼ�����Ĥͼ��������㣬��ö�ֵͼ�� */
	fillPoly(maskImage, maskVertexPoint, lineColor);
	Mat  maskedToDetectionImage;
	bitwise_and(todetectionImageEdges, maskImage, maskedToDetectionImage);

	/* ��ͼ�������̬ѧ�任���˲��Ȳ�������ֵ����Ȳ���������������������� */
	Mat morphologyMaskedToDetectionImage;
	Mat kernelDilate = getStructuringElement(MORPH_RECT, Size(5, 5));
	morphologyEx(maskedToDetectionImage, morphologyMaskedToDetectionImage, MORPH_DILATE, kernelDilate);
	Mat kernelErode = getStructuringElement(MORPH_RECT, Size(5, 5));
	morphologyEx(morphologyMaskedToDetectionImage, morphologyMaskedToDetectionImage, MORPH_ERODE, kernelErode);
	Mat kernelErodeAgain = getStructuringElement(MORPH_RECT, Size(5, 5));
	Mat morphologyMaskedToDetectionImageLast;
	morphologyEx(todetectionImageEdges, morphologyMaskedToDetectionImageLast, MORPH_DILATE, kernelErodeAgain);

	//subtract(morphologyMaskedToDetectionImage, morphologyMaskedToDetectionImageLast, morphologyMaskedToDetectionImage);
	/* ��һ������������������ */
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

	/* ���л���任������ȡÿһ���߶ε�б�ʺ�ÿһ���߶ζ˵�����꣬�����ҳ����߷ֱ���� */
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
			/* �������κβ��� */
			//lineDetectedInImage.erase(lineDetectedInImage.begin()+ laneLineIndex);

		}
	}

	/* �������任��ϳ���ÿһ���ߵ�б�ʣ��޳����Ե���Ⱥֵ������б�ʻ�����ͬ���߶�������������ϼ��㣬�����ҳ����߷ֱ�����޳� */
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
	�����������������ÿһ�������ʽ�����洢��vector������
param:
	laneLinePoint: Ҫ�����ĳ���������
return:
	������ĳ���������
*/
vector<Point> laneLineDetection::analysisLaneLinePoint(vector<Vec4f> laneLinePoint)
{
	vector<Point>analysisedLinePoint;
	/* ��ȡ���г������е�ÿһ�������ߵ����˶˵㣬�������е��������ڷ���ֵ�� */
	for (const auto& element : laneLinePoint)
	{
		analysisedLinePoint.push_back(Point(element[0], element[1]));
		analysisedLinePoint.push_back(Point(element[2], element[3]));
	}

	return analysisedLinePoint;
}

/*
illustrte:
	���ݼ�⵽�ĳ��������꣬������С���˷�������ϣ��õ����յĳ�����
param:
	lanelinePoint: ͼ���м�⵽�����ҳ����ߵ�����
	fittingOrder: ��������ϵĽ���
	fittingResult����������ϵĽ����ϵ������
return:
	��������ϵĽ����ϵ������
*/
vector<Point> laneLineDetection::laneLineCurveFitting(vector<Point> laneLinePoint, int fittingOrder, Mat toDetectionImage)
{
	/* ����opencv���Է�����⺯���������X */
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

	/*����opencv���Է�����⺯���������Y */
	cv::Mat inputArrayY = cv::Mat::zeros(fittingOrder + 1, 1, CV_64F);
	for (int i = 0; i < fittingOrder + 1; i++)
	{
		for (int k = 0; k < pointNumber; k++)
		{
			inputArrayY.at<double>(i, 0) = inputArrayY.at<double>(i, 0) +std::pow(laneLinePoint[k].x, i) * laneLinePoint[k].y;
		}
	}
	/* ����������ϵ������ */
	Mat fittingResultMatrix = Mat::zeros(fittingOrder + 1, 1, CV_64FC1);
	cv::solve(inputArrayX, inputArrayY, fittingResultMatrix, cv::DECOMP_LU);

	/* ����ϵ�����������Ϻ���������� */
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
	������ϵõ��ĳ���������㣬��ԭͼ�ϻ�����ʾ���ս��
param:
	toPlotImage: Ҫ���Ƴ����ߵ�ͼ��
	laneLinePoint�������߼������ս��
return:
	�������κ�ֵ
*/
void laneLineDetection::plotLaneLineDetectionResult(Mat toPlotImage, vector<Point>laneLinePoint)
{
	polylines(toPlotImage, laneLinePoint, false, cv::Scalar(0, 255, 255), 8, 8, 0);
}

laneLineDetection::~laneLineDetection()
{

}