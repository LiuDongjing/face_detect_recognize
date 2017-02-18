#include "Collector.h"
#include "toolkits.h"
bool Collector::collect(int label, double dist) {
	dists.push_back(dist);
	if(labels.size() < dists.size())
		labels.push_back(label);
	return true;
}
void Collector::init(size_t size) {
	assert(labels.size() == size);
}
Collector::Collector(string facedatabase) {
	FileStorage fs(facedatabase, FileStorage::READ);
	for (auto &e : fs["face_db"]) {
		Mat t;
		e >> t;
		facedb.push_back(t);
	}
	fs.release();
}
int Collector::getMostSimilar(Mat &e) {
	vector<int> index;
	sort(dists, index);
	e = facedb[index[0]].clone();
	return labels[index[0]];
}
vector<double> &Collector::getDists() {
	return dists;
}
vector<int> &Collector::getLabels() {
	return labels;
}
void Collector::clear() {
	dists.clear();
}