#pragma once
#include <opencv2/core/core.hpp>
#include <opencv2/face/facerec.hpp>
#include "Collector.h"
using namespace std;
using namespace cv;
using namespace face;
class FaceReg {
public:
	FaceReg(Ptr<FaceRecognizer> fr, int w, int h);
	void train(string dir);
	void test(string dir);
	string getMostSimilar(Mat &tar, Mat &sim);
	void load(string path);
	void save(string path);
private:
	Ptr<FaceRecognizer> model;
	Ptr<Collector> pc;
	int width;
	int height;
};

