#pragma once
#include <opencv2/core/core.hpp>
#include <opencv2/face/facerec.hpp>
using namespace std;
using namespace cv;
using namespace face;
class Collector :public PredictCollector {
public:
	Collector(string facedatabase);
	bool collect(int label, double dist) override;
	void init(size_t size) override;
	int getMostSimilar(Mat &e);
	vector<double> &getDists();
	vector<int> &getLabels();
	void clear();
private:
	vector<int> labels;
	vector<double> dists;
	vector<Mat> facedb;
};
