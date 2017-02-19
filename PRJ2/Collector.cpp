#include "Collector.h"
#include "toolkits.h"
bool Collector::collect(int label, double dist) {
	dists.push_back(dist);

	//labelsֻ��Ҫ��ʼ��һ��
	if(labels.size() < dists.size())
		labels.push_back(label);
	return true;
}
void Collector::init(size_t size) {
}
Collector::Collector(string facedatabase) {
	clog << "����ͼ���..." << endl << endl;
	FileStorage fs(facedatabase, FileStorage::READ);
	for (auto &e : fs["face_db"]) {
		Mat t;
		e >> t;
		facedb.push_back(t);
	}
	fs.release();
	clog << "�������!" << endl << endl;
}
int Collector::getMostSimilar(Mat &e) {
	vector<int> index;
	//dists��С�������У�index���ص���������ÿ��Ԫ����ԭ��dists�����λ��
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