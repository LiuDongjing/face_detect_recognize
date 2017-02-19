// ���ܽ�����ο������CommandLineParser������
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
		"{help||�˹��ߵ�ʹ��˵��}"
		"{face|../data/haarcascade_frontalface_alt.xml|Haar�������ģ��·��}"
		"{eye|../data/haarcascade_eye_tree_eyeglasses.xml|Haar���ۼ��ģ��·��}" 
		"{model|model.yaml|����(�����)ģ�͵�·��}"
		"{train||���ز���������ṩ����ѵ��ģ�͡�������ѵ����Ĳ��Խ��}"
		"{record||���ز���������ṩ�����ѳ������й�����չʾ��������Ƶ������record.avi��}"
		"{train_dir|../data/train|����������ѵ��������ͼ�񣬸�ʽname_id.jpg(pgm)����ͬ�����Ʊ�ʾͬһ����}"
		"{test_dir|../data/test|����������ѵ��������ͼ��}"
		"{method|eigenface|ʹ�õ�����ʶ���㷨��֧��eigen(eigenface)��fisher(fisherface)��lbph(LBPHface)��mylbph(����ʵ�ֵ�lbph�㷨)}"
		"{num|0|eigenface��fisher�㷨��component����������Ϊ0���㷨����������ֵ}"
		"{width|168|ͼ��ͳһ�����Ŀ��}"
		"{height|192|ͼ��ͳһ�����ĸ߶�}"
		"{src|0|��ƵԴ����������Ƶ·����Ҳ����������ͷ���}"
		"{radius|1|LBPH�㷨�е��ھӰ뾶}"
		"{neigh|8|LBPH�㷨�е��ھ�����}"
		"{nx|8|LBPH�㷨��ˮƽ����ָ�����}"
		"{ny|8|LBPH�㷨�д�ֱ����ָ�����}"
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
			cerr << "��֧�ֵķ���: " << method << endl;
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
		cerr << "�޷������(����Ƶ)!" << endl;
		return 0;
	}
	if (rec) {
		record.open("record.avi", VideoWriter::fourcc('D', 'I', 'V', 'X') , 30,
			Size(cap.get(CAP_PROP_FRAME_WIDTH), cap.get(CAP_PROP_FRAME_HEIGHT)));
		if (!record.isOpened()) {
			cerr << "�޷�����¼����Ƶ��" << endl;
			return 0;
		}
	}
	FaceDetector fd;
	if (!fd.load(facePath, eyePath)) {
		cerr << "�޷������������ģ��!" << endl;
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
		cerr << "��֧�ֵķ���: " << method << endl;
		return -1;
	}
	model->load(modelPath);
	Mat frame;
	cap >> frame;
	namedWindow("���");
	while (!frame.empty()) {
		vector<Mat> aligned;
		vector<Rect> pos;
		if (fd.getFaceRect(frame, pos, aligned)) {
			Mat prof;
			//�˴����ص�����һ������Ч��
			string name = model->getMostSimilar(aligned[0], prof);
			mark(frame, pos[0], name, prof);
			if (rec)
				for(int i = 0; i < 60; i++)
					record << frame;
			imshow("���", frame);
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
		imshow("���", frame);
		if (rec)
			record << frame;
		waitKey(1000 / 100);
		cap >> frame;
	}
	if (rec)
		record.release();
	return 0;
}