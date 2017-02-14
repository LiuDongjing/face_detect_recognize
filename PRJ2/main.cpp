#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/face/facerec.hpp>
#include "FaceDetector.h"
#include "toolkits.h"
#include "FaceDetector.h"
#include "EigenFace.h"
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
		train(parser);
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
	FaceRegBase *model = nullptr;
	if (method == "eigenface") {
		model = new EigenFace();
	}
	model->load(modelPath);

	vector<string> label2name, sample2path;
	vector<int> sample2label = model->getLabels();
	if (!configure(confPath, label2name, sample2path)) {
		cout << "Configure failed!" << endl;
		return 0;
	}
	Mat frame;
	cap >> frame;
	namedWindow("camera");
	while (!frame.empty()) {
		vector<Mat> aligned;
		vector<Rect> pos;
		if (fd.getFaceRect(frame, pos, &aligned)) {
			int label, sampleNum;
			double dist;
			label = model->predict(aligned[0], sampleNum, dist);
			string name;
			Mat prof;
			if (label < 0) {
				name = "unknown";
				mark(frame, pos[0], name, prof, false);
			}
			else {
				name = label2name[label];
				prof = imread(sample2path[sampleNum]);
				mark(frame, pos[0], name, prof);
			}
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