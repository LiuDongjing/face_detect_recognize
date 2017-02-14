#pragma once
#include "FaceRegBase.h"
class FisherFace :public FaceRegBase {
public:
	FisherFace();
	bool load(string path) override;
	void save(string path)  override;
	void train(vector<Mat> &samples, vector<int> &labels)  override;
	int predict(Mat &test, int &sampleNum, double &dist, vector<int> &index = vector<int>())  override;
	double threshold()  override;
	void threshold(double th)  override;
	vector<int> getLabels() override;
private:
	Ptr<BasicFaceRecognizer> model;

};