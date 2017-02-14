#pragma once
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/face/facerec.hpp>
using namespace std;
using namespace cv;
using namespace face;
class FaceRegBase {
public:
	FaceRegBase();
	virtual bool load(string path);
	virtual void save(string path);
	virtual void train(vector<Mat> &samples, vector<int> &labels);
	virtual int predict(Mat &test, int &sampleNum, double &dist, vector<int> &index = vector<int>());
	virtual double threshold();
	virtual void threshold(double th);
	virtual vector<int> getLabels();
};