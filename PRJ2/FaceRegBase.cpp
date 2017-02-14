#include "FaceRegBase.h"
FaceRegBase::FaceRegBase() {}
bool FaceRegBase::load(string path) { return false; }
void FaceRegBase::save(string path) {}
void FaceRegBase::train(vector<Mat> &samples, vector<int> &labels) {}
int FaceRegBase::predict(Mat &test, int &sampleNum, double &dist, vector<int> &index) { return -1; }
double FaceRegBase::threshold() { return -1; }
void FaceRegBase::threshold(double th) {}
vector<int> FaceRegBase::getLabels() { return vector<int>(); }