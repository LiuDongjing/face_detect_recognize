#include "Collector.h"
#include "toolkits.h"
bool Collector::collect(int label, double dist) {
	dists.push_back(dist);

	//labels只需要初始化一次
	if(labels.size() < dists.size())
		labels.push_back(label);
	return true;
}
void Collector::init(size_t size) {
}
Collector::Collector(string facedatabase) {
	clog << "读入图像库..." << endl << endl;
	FileStorage fs(facedatabase, FileStorage::READ);
	for (auto &e : fs["face_db"]) {
		Mat t;
		e >> t;
		facedb.push_back(t);
	}
	fs.release();
	clog << "读入完毕!" << endl << endl;
}
int Collector::getMostSimilar(Mat &e) {
	vector<int> index;
	//dists从小到大排列，index返回的是排序后的每个元素在原来dists数组的位置
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