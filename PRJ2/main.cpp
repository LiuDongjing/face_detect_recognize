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
		"{help||usages about this tool.}"
		"{face|../data/haarcascade_frontalface_alt.xml|}"
		"{eye|../data/haarcascade_eye_tree_eyeglasses.xml|}" 
		"{model|model.yaml|}"
		"{train||}"
		"{record||}"
		"{train_dir|../data/train|}"
		"{test_dir|../data/test|}"
		"{method|eigenface|}"
		"{num|10|}"
		"{threshold|400|}"
		"{w|168|}"
		"{h|192|}"
		"{conf|conf.yaml|}"
		"{src|0|}"
	);
	if (parser.has("help")) {
		parser.printMessage();
	}
	if (parser.has("train")) {
		string traindir = parser.get<string>("train_dir");
		string testdir = parser.get<string>("test_dir");
		string modelPath = parser.get<string>("model");
		string method = parser.get<string>("method");
		
		int w = parser.get<int>("w");
		int h = parser.get<int>("h");
		int num = parser.get<int>("num");
		double threshold = parser.get<double>("threshold");
		if (!parser.check()) {
			parser.printErrors();
			return -1;
		}
		transform(method.begin(), method.end(), method.begin(), tolower);
		FaceReg *model = nullptr;
		if (method == "eigenface") {
			model = new FaceReg(createEigenFaceRecognizer(), w, h);
		}
		else if (method == "lbph") {
			model = new FaceReg(createLBPHFaceRecognizer(), w, h);
		}
		else if (method == "fisher") {
			model = new FaceReg(createFisherFaceRecognizer(), w, h);
		}
		else if (method == "mylbph") {
			model = new FaceReg(createMyLBPHFaceRecognizer(), w, h);
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
	string confPath = parser.get<string>("conf");
	string modelPath = parser.get<string>("model");
	string src = parser.get<string>("src");
	string method = parser.get<string>("method");
	VideoWriter record;
	bool rec = parser.has("record");
	int w = parser.get<int>("w");
	int h = parser.get<int>("h");
	if (!parser.check()) {
		parser.printErrors();
		return 0;
	}
	VideoCapture cap;
	if (isdigit(src[0]))
		cap.open(src[0] - '0');
	else
		cap.open(src);
	if (!cap.isOpened()) {
		cerr << "Open video or camera failed!" << endl;
		return 0;
	}
	if (rec) {
		record.open("record.avi", VideoWriter::fourcc('D', 'I', 'V', 'X') , 30,
			Size(cap.get(CAP_PROP_FRAME_WIDTH), cap.get(CAP_PROP_FRAME_HEIGHT)));
		if (!record.isOpened()) {
			cerr << "open record video failed!" << endl;
			return 0;
		}
	}
	FaceDetector fd;
	if (!fd.load(facePath, eyePath)) {
		cerr << "Load face detect model failed!" << endl;
		return 0;
	}
	FaceReg *model = nullptr;
	if (method == "eigenface") {
		model = new FaceReg(createEigenFaceRecognizer(), w, h);
	}
	else if (method == "lbph") {
		model = new FaceReg(createLBPHFaceRecognizer(), w, h);
	}
	else if (method == "fisher") {
		model = new FaceReg(createFisherFaceRecognizer(), w, h);
	}
	else if (method == "mylbph") {
		model = new FaceReg(createMyLBPHFaceRecognizer(), w, h);
	}
	else {
		cerr << "不支持的方法: " << method << endl;
		return -1;
	}
	model->load(modelPath);
	Mat frame;
	cap >> frame;
	namedWindow("camera");
	Collector cc = Collector("face_database.yaml");
	while (!frame.empty()) {
		vector<Mat> aligned;
		vector<Rect> pos;
		if (fd.getFaceRect(frame, pos, &aligned)) {
			int label, sampleNum;
			double dist;
			Mat prof;

			//此处返回的名字一定是有效的
			string name = model->getMostSimilar(aligned[0], prof);
			mark(frame, pos[0], name, prof);
			if (rec)
				for(int i = 0; i < 60; i++)
					record << frame;
			imshow("camera", frame);
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
		imshow("camera", frame);
		if (rec)
			record << frame;
		waitKey(1000 / 100);
		cap >> frame;
	}
	if (rec)
		record.release();
	return 0;
}