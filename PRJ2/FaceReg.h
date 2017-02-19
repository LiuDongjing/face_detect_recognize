/*统一管理训练、测试和识别过程的类。每个人脸识别算法都继承了FaceRecognizer。
  FaceReg通过传入的FaceRecognizer对象，调用它的train和predict方法，完成不同
  下的训练、测试和识别过程*/
#pragma once
#include <opencv2/core/core.hpp>
#include <opencv2/face/facerec.hpp>
#include "Collector.h"
using namespace std;
using namespace cv;
using namespace face;
class FaceReg {
public:
	/*fr是识别算法实例；w和h分别是预处理后图像的长和宽；facePath是
	人脸库序列化后保存路径*/
	FaceReg(Ptr<FaceRecognizer> fr, int w, int h);
	FaceReg(Ptr<FaceRecognizer> fr, int w, int h, string facePath);

	/*训练模型；dir包含了训练的原始图像集；facePath同上*/
	void train(string dir, string facePath = "face_database.yaml");

	/*测试模型；dir包含了测试的原始图像集；facePath同上*/
	void test(string dir, string facePath = "face_database.yaml");

	/*获取库中与tar最像的人脸sim，该人脸的名称经返回值得到；facePath同上*/
	string getMostSimilar(Mat &tar, Mat &sim, string facePath = "face_database.yaml");

	/*加载模型*/
	void load(string path);

	/*保存模型*/
	void save(string path);
private:
	Ptr<FaceRecognizer> model;
	Ptr<Collector> pc;
	int width;
	int height;
};

