#include "EigenFace.h"
#include "toolkits.h"
#include <vector>
EigenFace::EigenFace() {
	model = createEigenFaceRecognizer(100);
}
bool EigenFace::load(string path) {
	FileStorage fs(path, FileStorage::READ);
	if (!fs.isOpened()) {
		cerr << "Open " << path << "failed!" << endl;
		return false;
	}
	fs["eigvec"] >> eigvec;
	sample2label.clear();
	for (auto &e : fs["sample2label"])
	{
		int k;
		e >> k;
		sample2label.push_back(k);
	}
	proj.clear();
	for (auto &e : fs["proj"]) {
		Mat t;
		e >> t;
		proj.push_back(t);
	}
	fs["mean"] >> mean;
	fs["thresh"] >> thresh;
	fs["faceSize"] >> faceSize;
	fs.release();
	return true;
}
void EigenFace::save(string path) {
	FileStorage fs(path, FileStorage::WRITE);
	fs << "eigvec" << eigvec;
	fs << "sample2label" << "[";
	for (auto &e : sample2label) fs << e;
	fs << "]";
	fs << "proj" << "[";
	for (auto &e : proj) fs << e;
	fs << "]";
	fs << "mean" << mean;
	fs << "thresh" << thresh;
	fs << "faceSize" << faceSize;
	fs.release();
}
void EigenFace::train(vector<Mat> &samples, vector<int> &labels) {
	faceSize = Size(samples[0].cols, samples[0].rows);
	model->train(samples, labels);
	sample2label = labels;
	eigvec = model->getEigenVectors();
	proj = model->getProjections();
	mean = model->getMean();
	thresh = model->getThreshold();
}
int EigenFace::predict(Mat &test, int &sampleNum, double &dist, vector<int> &index) {
	Mat x = test.clone();
	preprocess(x, faceSize.width, faceSize.height, true);
	x -= mean;
	Mat p = x * eigvec;
	vector<double> dists;
	for (auto&e : proj)
		dists.push_back(norm(e, p));
	sort(dists, index);
	if (dists[0] <= thresh) {
		sampleNum = index[0];
		dist = dists[0];
		return sample2label[sampleNum];
	}
	else
		return -1;
}
double EigenFace::threshold() {
	return thresh;
}
void EigenFace::threshold(double th) {
	thresh = th;
	model->setThreshold(thresh);
}
vector<int> EigenFace::getLabels() { return sample2label; }