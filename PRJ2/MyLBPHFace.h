/*
  自己实现的简单版本的LBPH算法。邻居固定为8个，半径为1；
  划分人脸的区域个数可变。没有使用uniform pattern。
*/
#pragma once
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/face/facerec.hpp>
#include <iostream>
using namespace std;
using namespace cv;
using namespace face;
class MyLBPHFace : public FaceRecognizer {
public :
	/*x、y分别是图像横向和纵向分割的数量；默认均是5*/
	MyLBPHFace();
	MyLBPHFace(int x, int y);
	double getThreshold() const override;
	void load(const FileStorage &fs) override;
	void predict(InputArray src, Ptr<PredictCollector> collector) const override;
	void save(FileStorage &fs) const override;
	void train(InputArrayOfArrays src, InputArray labels) override;
	void setThreshold(double thresh) override;
	String getLabelInfo(int label) const  override;
	void setLabelInfo(int label, const String &strInfo) override;
private:
	/*获取img的特征向量vec*/
	void getDescVec(Mat &img, Mat &vec) const;

	/*将img每个像素按LBP的方式编码*/
	void encode(Mat &img) const;

	/*获取编码后图像的直方图h*/
	void hist(Mat &img, Mat &h) const;
private:
	Mat desc;//NxM的矩阵，N是人脸的数量，M是LBPH描述符的大小
	vector<int> lab;//N个人脸对应的标签
	int nX = 8; //将一张图像水平方向上分成nX份
	int nY = 8; //将图像水平分成nY份
	double radius = 1; //邻居半径，目前未使用
	int num = 8;// 邻居的数量，目前未使用
	double threshold;
	vector<string> labInfo;
};

/*实例化一个MyLBPHFace，x、y分别是图像水平和竖直方向分割分数*/
Ptr<MyLBPHFace> createMyLBPHFaceRecognizer();
Ptr<MyLBPHFace> createMyLBPHFaceRecognizer(int x, int y);