/*基于Haar的级联检测器。可用于检测人脸和人眼。
  使用的是已经训练好的模型*/
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

	//加载人脸和人眼检测模型，加载成功返回true，否则false
	bool load(string facePath, string eyePath);

	/*获取img中的人脸区域result，以及校正后的人脸数据；默认最多检测一个对象*/
	bool getFaceRect(Mat &img, vector<Rect> &result, 
		vector<Mat> &aligned, size_t len = 1);
private:
	//用model检测grayImg，检测结果保存在result中，其余是检测参数
	void detect(CascadeClassifier &model, Mat &grayImg, vector<Rect> &result, 
		double scale = 1.5, int minNeibor = 3, Size minSize = Size(5, 5), 
		Size maxSize = Size());

	/*人脸img绕双眼中心center旋转至水平后，按双眼间距一半r按比例
	裁剪人脸区域;具体的比例参考private成员变量*/
	Rect cropFace(Mat &img, Point center, double r);

	/*在img上框出裁剪后人脸未旋转前的位置*/
	void drawCrop(Mat &img, RotatedRect &rot, Scalar color = Scalar(255, 0, 0));

	double scaleUp = 1.6;//以双眼中心向上裁剪比例
	double scaleDown = 3.0;//以双眼中心向下裁剪比例
	double scaleHoriz = 1.8;//以双眼中心水平裁剪比例

	CascadeClassifier face;
	CascadeClassifier eye;
};