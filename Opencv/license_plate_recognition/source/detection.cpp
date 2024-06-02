#include<iostream>
#include<string>
#include<vector>
#include<opencv2/opencv.hpp>
#include"detection.h"

using namespace std;
using namespace cv;

LicensePlateNumberDetection::LicensePlateNumberDetection()
{

}

LicensePlateNumberDetection::~LicensePlateNumberDetection()
{

}

/*
illustrate:
	��ȡҪʶ��ĳ���ͼ��
param:
	imagePath: ͼ��洢·��
return��
	None
*/
Mat LicensePlateNumberDetection::readDetectionImage(string imagePath)
{
	Mat toDetetionImage;
	toDetetionImage = imread(imagePath);
	return toDetetionImage;
}

/*
illustrate:
	�Կ����ǳ��Ƶ�λ�ý��г��Զ�λ
param:
	toLocateImage: Ҫ��λ��ͼ��
return��
	vector�������洢���ƶ�λ����Ӿ��ο�
*/
vector<Rect> LicensePlateNumberDetection::licensePlateLocate(cv::Mat toLocateImage)
{
	/* ������̬ѧ����Ե�����������ͼ����г�������*/
	Mat toLocateImageGray,imageGaussBlur, imageOpenOperation, imageThreshhold, imageCannyEdge, locatedImage;
	cvtColor(toLocateImage, toLocateImageGray, COLOR_BGR2GRAY);
	GaussianBlur(toLocateImageGray, imageGaussBlur, Size(9,9), 0, 0);
	Mat element = getStructuringElement(MORPH_RECT, Size(21, 21));
	morphologyEx(imageGaussBlur, imageOpenOperation, MORPH_OPEN, element);
	addWeighted(imageGaussBlur, 1, imageOpenOperation, -1, 0, imageOpenOperation);
	threshold(imageOpenOperation, imageThreshhold, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
	Canny(imageThreshhold, imageCannyEdge, 250, 200);
	element = getStructuringElement(MORPH_RECT, Size(25,25));
	morphologyEx(imageCannyEdge, imageCannyEdge, MORPH_CLOSE, element);
	morphologyEx(imageCannyEdge, imageCannyEdge, MORPH_OPEN, element);
	/* �ҳ����ܴ��ڳ��Ƶ�λ�� */
	vector<vector<Point>>contours;
	vector<Vec4i>hierarchy;
	findContours(imageCannyEdge, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point());
	Scalar color(0, 0, 255);
	vector<Rect>possibleRectangle;
	for (int indexContours = 0; indexContours < contours.size(); indexContours++)
	{
		RotatedRect numberRect = minAreaRect(contours[indexContours]);
		Point2f rectPoint[4];
		numberRect.points(rectPoint);
		float width = rectPoint[0].x - rectPoint[1].x;
		float height = rectPoint[0].y - rectPoint[3].y;
		if (width < height)
		{
			float temp = width;
			width = height;
			height = temp;
		}
		float ratio = width / height;
		if (ratio <0.5)
		{
			Rect rectBox = numberRect.boundingRect();
			rectBox.x = rectBox.x-2;
			rectBox.y = rectBox.y - 2 ;
			rectBox.width = rectBox.width+3;
			rectBox.height = rectBox.height-23;
			rectangle(toLocateImage, rectBox, Scalar(0, 0, 255), 2);
			possibleRectangle.push_back(rectBox);
		}
	}
	return possibleRectangle;
}


/*
illustrate:
	�Զ�λ���Ŀ����ǳ��Ƶ�������зָ��ȡ�����ַ�
param:
	toLocateImage: Ҫ��λ��ͼ��
	possibleRectangle��vector���ͣ��洢��λ���Ŀ��ܵĳ������������
return��
	vector�������洢�ָ�õ����ַ�����ͼ��
*/
vector<Mat> LicensePlateNumberDetection::licensePlateSegment(std::vector<cv::Rect>possibleRectangle, Mat toDetectionImage)
{
	/* �Զ�λ����λ�ý��зָ�����վ��ο����ʽ��ȡ���� */
	Mat toDetectionImageGray;
	cvtColor(toDetectionImage,toDetectionImageGray,COLOR_BGR2GRAY);
	vector<Mat>possibleCharactor;
	for(Rect elem: possibleRectangle)
	{
		Mat locatedToDetectionImage;
		locatedToDetectionImage = toDetectionImageGray(elem);
		resize(locatedToDetectionImage, locatedToDetectionImage, Size(440, 140));
		GaussianBlur(locatedToDetectionImage, locatedToDetectionImage, Size(7, 7), 0, 0);
		Canny(locatedToDetectionImage, locatedToDetectionImage, 80, 60);
		Mat kernelDilate = getStructuringElement(MORPH_RECT, Size(5, 5));
		morphologyEx(locatedToDetectionImage, locatedToDetectionImage, MORPH_DILATE, kernelDilate);
		Mat kernelErode = getStructuringElement(MORPH_RECT, Size(2, 2));
		morphologyEx(locatedToDetectionImage, locatedToDetectionImage, MORPH_ERODE, kernelErode);
		vector<vector<Point>> contours;
		vector<Vec4i> hierarchy;
		findContours(locatedToDetectionImage, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
		if (contours.size() == 0)
		{
			continue;
		}
		drawContours(locatedToDetectionImage, contours, -1, Scalar(255, 255, 255),FILLED);
		Mat kernelOpen = getStructuringElement(MORPH_RECT, Size(9,11));
		morphologyEx(locatedToDetectionImage, locatedToDetectionImage, MORPH_OPEN, kernelOpen);
		Mat kernelErodelast = getStructuringElement(MORPH_RECT, Size(7, 7));
		morphologyEx(locatedToDetectionImage, locatedToDetectionImage, MORPH_ERODE, kernelErodelast);
		Mat locatChinese;
		Mat kernelDilatelast = getStructuringElement(MORPH_RECT, Size(13, 13));
		morphologyEx(locatedToDetectionImage, locatChinese, MORPH_DILATE, kernelDilatelast);
		/* �Ƚ����к��ֵ�ͼƬ����vector�� */
		vector<vector<Point>> contours3;
		vector<Vec4i> hierarchy3;
		findContours(locatChinese, contours3, hierarchy3, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
		if (contours3.size() == 0)
		{
			continue;
		}
		for (int contoursIndex = 0; contoursIndex < contours3.size(); contoursIndex++)
		{
			Rect rectangleBouding = boundingRect(contours3[contoursIndex]);
			Mat charactorChinese = locatedToDetectionImage(rectangleBouding);
			possibleCharactor.push_back(charactorChinese);
		}
		/* �ٽ���ĸ�����ַ���vector�� */
		vector<vector<Point>> contours2;
		vector<Vec4i> hierarchy2;
		findContours(locatedToDetectionImage, contours2, hierarchy2, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
		if (contours2.size()==0)
		{
			continue;
		}
		for (int contoursIndex = 0; contoursIndex < contours2.size(); contoursIndex++)
		{
			Rect rectangleBouding = boundingRect(contours2[contoursIndex]);
			Mat charactor = locatedToDetectionImage(rectangleBouding);
			possibleCharactor.push_back(charactor);
		}
	}
	return possibleCharactor;
}

/*
illustrate:
	����ȡ���ĳ����ַ�����ʶ�𣬵õ�������ַ������
param:
	toRecongnizeImage: vector���ͣ��洢�ָ�õ��ĳ��ƺ��ַ���
	
return��
	stringle���ͣ����յ�ʶ����
*/
string LicensePlateNumberDetection::licensePlateRecongnize(vector<Mat>toRecongnizeImage)
{
	string license;
	/* ��ʶ����*/

	/* ��ʶ����ĸ������*/
	string a="sb";
	return a;
}