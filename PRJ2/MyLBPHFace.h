#pragma once
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/face/facerec.hpp>
#include <iostream>
// to-do: 可变半径邻居；规范化的LBP；划分不同的cell；knn分类；
using namespace std;
using namespace cv;
using namespace face;
class MyLBPHFace : public FaceRecognizer {
public :
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
	void getDescVec(Mat &img, Mat &vec) const;
	void encode(Mat &img) const;
	void hist(Mat &img, Mat &h) const;
private:
	Mat desc;//NxM的矩阵，N是人脸的数量，M是LBPH描述符的大小
	vector<int> lab;//N个人脸对应的标签
	int nX = 5; //将一张图像水平方向上分成nX份
	int nY = 5; //将图像水平分成nY份
	double radius = 1; //邻居半径，目前未使用
	int num = 8;// 邻居的数量，目前未使用
	double threshold;
	vector<string> labInfo;
};
Ptr<MyLBPHFace> createMyLBPHFaceRecognizer();
Ptr<MyLBPHFace> createMyLBPHFaceRecognizer(int x, int y);