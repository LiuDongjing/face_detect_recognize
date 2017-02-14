#pragma once
#include "FaceRegBase.h"
class EigenFace :public FaceRegBase {
public:
	EigenFace();
	bool load(string path) override;
	void save(string path)  override;
	void train(vector<Mat> &samples, vector<int> &labels)  override;
	int predict(Mat &test, int &sampleNum, double &dist, vector<int> &index = vector<int>())  override;
	double threshold()  override;
	void threshold(double th)  override;
	vector<int> getLabels() override;
private:
	Ptr<BasicFaceRecognizer> model;
	Mat eigvec;
	vector<Mat> proj;
	Mat mean;
	vector<int> sample2label;
	double thresh;
	Size faceSize;
};
