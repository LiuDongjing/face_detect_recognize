/*
	继承了PredictCollector，重写了collect方法。
 在调用模型的predict(InputArray, PredictCollector)时会收集预测过程中的数据。
	eigenface、fisherface和lbphface的predict(InputArray, PredictCollector)方法，
 会依次(顺序和train方法传入的人脸vector一致)遍历保存的人脸特征向量，计算该向量
 与输入人脸的特征向量之间的距离dist，然后该向量对应的label和dist传入PredictCollector
 的collect方法。
	因此PredictCollector可以将输入人脸与库中每张人脸的距离已经库中
 人脸的label保存下来。而下面的Collector方法正是这么做的。保存了上述两个数据，就
 可以用kNN做人脸分类。
*/
#pragma once
#include <opencv2/core/core.hpp>
#include <opencv2/face/facerec.hpp>
using namespace std; 
using namespace cv;
using namespace face;
class Collector :public PredictCollector {
public:
	/* 人脸数据库路径。因为getMostSimilar方法要返回在
	   库中与输入人脸最像的人脸图像。*/
	Collector(string facedatabase);

	/*由模型的predict方法调用；注意这里的label并不是预测得到的label，而是
	  库中人脸的label;该方法会把待预测人脸与每个库中人脸的距离都保存下来*/
	bool collect(int label, double dist) override;
	void init(size_t size) override;

	/*返回本次预测中与待预测人脸最像的人脸。人脸的图像通过e返回，人脸的label
	  通过返回值得到*/
	int getMostSimilar(Mat &e);

	/*返回此次预测collect得到的距离向量*/
	vector<double> &getDists();

	/*返回库中人脸的label*/
	vector<int> &getLabels();

	/*清空已保存的数据；注意只会清空dists数组，因为其他的数据每次预测都是一样的*/
	void clear();
private:
	vector<int> labels;
	vector<double> dists;
	vector<Mat> facedb;
};
