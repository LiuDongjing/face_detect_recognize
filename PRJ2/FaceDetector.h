#pragma once
#include <iostream>
#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
using namespace std;
using namespace cv;
class FaceDetector {
public:
	FaceDetector();
	~FaceDetector();
	bool load(string facePath, string eyePath);
	bool getFaceRect(Mat &img, vector<Rect> &result, vector<Mat> *alignedFace = nullptr, size_t len = 1);
private:
	void detect(CascadeClassifier &model, Mat &grayImg, vector<Rect> &result, 
		double scale = 1.5, int minNeibor = 3, Size minSize = Size(5, 5), 
		Size maxSize = Size());
	Rect cropFace(Mat &img, Point center, double r);
	void drawCrop(Mat &img, RotatedRect &rot, Scalar color = Scalar(255, 0, 0));

	double scaleUp = 1.6;
	double scaleDown = 3.0;
	double scaleHoriz = 1.8;

	CascadeClassifier face;
	CascadeClassifier eye;
};