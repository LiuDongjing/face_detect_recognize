#include "FisherFace.h"
FisherFace::FisherFace() {
	model = createFisherFaceRecognizer();
}
bool FisherFace::load(string path){
	model->load(path);
	return true;
}
void FisherFace::save(string path) {
	model->save(path);
}
void FisherFace::train(vector<Mat> &samples, vector<int> &labels) {
	model->train(samples, labels);
}
int FisherFace::predict(Mat &test, int &sampleNum, double &dist, vector<int> &index) {
	return -1;
}
double FisherFace::threshold() { return 0; }
void FisherFace::threshold(double th) {}
vector<int> FisherFace::getLabels() { return vector<int>(); }