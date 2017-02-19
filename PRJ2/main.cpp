// 功能介绍请参考下面的CommandLineParser的配置
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/face/facerec.hpp>
#include "FaceDetector.h"
#include "toolkits.h"
#include "FaceDetector.h"
#include "FaceReg.h"
#include "Collector.h"
#include "MyLBPHFace.h"
using namespace std;
using namespace cv;
using namespace face;
int main(int argc, char *argv[]) {
	CommandLineParser parser(argc, argv, 
		"{help||此工具的使用说明}"
		"{face|../data/haarcascade_frontalface_alt.xml|Haar人脸检测模型路径}"
		"{eye|../data/haarcascade_eye_tree_eyeglasses.xml|Haar人眼检测模型路径}" 
		"{model|model.yaml|保存(或加载)模型的路径}"
		"{train||开关参数。如果提供，则训练模型。并呈现训练后的测试结果}"
		"{record||开关参数，如果提供，则会把程序运行过程中展示出来的视频保存至record.avi中}"
		"{train_dir|../data/train|包含了用于训练的人脸图像，格式name_id.jpg(pgm)。相同的名称表示同一个人}"
		"{test_dir|../data/test|包含了用于训练的人脸图像}"
		"{method|eigenface|使用的人脸识别算法，支持eigen(eigenface)、fisher(fisherface)、lbph(LBPHface)和mylbph(本人实现的lbph算法)}"
		"{num|0|eigenface和fisher算法中component的数量，置为0由算法决定具体数值}"
		"{width|168|图像统一处理后的宽度}"
		"{height|192|图像统一处理后的高度}"
		"{src|0|视频源。可以是视频路径，也可以是摄像头编号}"
		"{radius|1|LBPH算法中的邻居半径}"
		"{neigh|8|LBPH算法中的邻居数量}"
		"{nx|8|LBPH算法中水平方向分割数量}"
		"{ny|8|LBPH算法中垂直方向分割数量}"
	);
	if (parser.has("help")) {
		parser.printMessage();
	}
	string modelPath = parser.get<string>("model");
	string method = parser.get<string>("method");

	int w = parser.get<int>("width");
	int h = parser.get<int>("height");
	int num = parser.get<int>("num");
	int radius = parser.get<int>("radius");
	int neigh = parser.get<int>("neigh");
	int nx = parser.get<int>("nx");
	int ny = parser.get<int>("ny");
	if (parser.has("train")) {
		string traindir = parser.get<string>("train_dir");
		string testdir = parser.get<string>("test_dir");
		
		if (!parser.check()) {
			parser.printErrors();
			return -1;
		}
		transform(method.begin(), method.end(), method.begin(), tolower);
		Ptr<FaceReg> model;
		if (method == "eigen") {
			model = new FaceReg(createEigenFaceRecognizer(num), w, h);
		}
		else if (method == "lbph") {
			model = new FaceReg(createLBPHFaceRecognizer(radius, neigh, nx, ny), w, h);
		}
		else if (method == "fisher") {
			model = new FaceReg(createFisherFaceRecognizer(num), w, h);
		}
		else if (method == "mylbph") {
			model = new FaceReg(createMyLBPHFaceRecognizer(nx, ny), w, h);
		}
		else {
			cerr << "不支持的方法: " << method << endl;
			return -1;
		}
		model->train(traindir);
		model->test(testdir);
		model->save(modelPath);
		return 0;
	}
	string facePath = parser.get<string>("face");
	string eyePath = parser.get<string>("eye");
	string src = parser.get<string>("src");
	VideoWriter record;
	bool rec = parser.has("record");
	if (!parser.check()) {
		parser.printErrors();
		return 0;
	}
	transform(method.begin(), method.end(), method.begin(), tolower);
	VideoCapture cap;
	if (isdigit(src[0]))
		cap.open(src[0] - '0');
	else
		cap.open(src);
	if (!cap.isOpened()) {
		cerr << "无法打开相机(或视频)!" << endl;
		return 0;
	}
	if (rec) {
		record.open("record.avi", VideoWriter::fourcc('D', 'I', 'V', 'X') , 30,
			Size(cap.get(CAP_PROP_FRAME_WIDTH), cap.get(CAP_PROP_FRAME_HEIGHT)));
		if (!record.isOpened()) {
			cerr << "无法保存录制视频！" << endl;
			return 0;
		}
	}
	FaceDetector fd;
	if (!fd.load(facePath, eyePath)) {
		cerr << "无法加载人脸检测模型!" << endl;
		return 0;
	}
	Ptr<FaceReg> model;
	if (method == "eigen") {
		model = new FaceReg(createEigenFaceRecognizer(num), w, h, "face_database.yaml");
	}
	else if (method == "lbph") {
		model = new FaceReg(createLBPHFaceRecognizer(radius, neigh, nx, ny), w, h, "face_database.yaml");
	}
	else if (method == "fisher") {
		model = new FaceReg(createFisherFaceRecognizer(num), w, h, "face_database.yaml");
	}
	else if (method == "mylbph") {
		model = new FaceReg(createMyLBPHFaceRecognizer(nx, ny), w, h, "face_database.yaml");
	}
	else {
		cerr << "不支持的方法: " << method << endl;
		return -1;
	}
	model->load(modelPath);
	Mat frame;
	cap >> frame;
	namedWindow("相机");
	while (!frame.empty()) {
		vector<Mat> aligned;
		vector<Rect> pos;
		if (fd.getFaceRect(frame, pos, aligned)) {
			Mat prof;
			//此处返回的名字一定是有效的
			string name = model->getMostSimilar(aligned[0], prof);
			mark(frame, pos[0], name, prof);
			if (rec)
				for(int i = 0; i < 60; i++)
					record << frame;
			imshow("相机", frame);
			char c;
			do {
				c = waitKey();
			} while (' ' != c && c != 'c');
			if (c == 'c')
				break;
			cap >> frame;
			continue;
		}
		else if (!pos.empty()) {
			mark(frame, pos[0], "unknown", Mat(), false);
		}
		imshow("相机", frame);
		if (rec)
			record << frame;
		waitKey(1000 / 100);
		cap >> frame;
	}
	if (rec)
		record.release();
	return 0;
}