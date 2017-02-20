/*����Haar�ļ���������������ڼ�����������ۡ�
  ʹ�õ����Ѿ�ѵ���õ�ģ��*/
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

	//�������������ۼ��ģ�ͣ����سɹ�����true������false
	bool load(string facePath, string eyePath);

	/*��ȡimg�е���������result���Լ�У������������ݣ�Ĭ�������һ������*/
	bool getFaceRect(Mat &img, vector<Rect> &result, 
		vector<Mat> &aligned, size_t len = 1);
private:
	//��model���grayImg�������������result�У������Ǽ�����
	void detect(CascadeClassifier &model, Mat &grayImg, vector<Rect> &result, 
		double scale = 1.5, int minNeibor = 3, Size minSize = Size(5, 5), 
		Size maxSize = Size());

	/*����img��˫������center��ת��ˮƽ�󣬰�˫�ۼ��һ��r������
	�ü���������;����ı����ο�private��Ա����*/
	Rect cropFace(Mat &img, Point center, double r);

	/*��img�Ͽ���ü�������δ��תǰ��λ��*/
	void drawCrop(Mat &img, RotatedRect &rot, Scalar color = Scalar(255, 0, 0));

	double scaleUp = 1.6;//��˫���������ϲü�����
	double scaleDown = 3.0;//��˫���������²ü�����
	double scaleHoriz = 1.8;//��˫������ˮƽ�ü�����

	CascadeClassifier face;
	CascadeClassifier eye;
};