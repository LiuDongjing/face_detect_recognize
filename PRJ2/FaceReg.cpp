#include "FaceReg.h"
#include "toolkits.h"
#include "Collector.h"
#include <set>
#include <vector>
#include <experimental/filesystem>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
namespace fs = std::experimental::filesystem;
using namespace fs;
FaceReg::FaceReg(Ptr<FaceRecognizer> fr, int w, int h) {
	model = fr;
	pc = new Collector("face_database.yaml");
	width = w;
	height = h;
}
void FaceReg::train(string dir) {
	vector<string> label2name;
	set<string> nameGot;
	vector<Mat> imgs;
	map<string, int> name2label;
	vector<int> labels;
	int label = 0;
	cout << "+----------Begin Train----------+" << endl;
	cout << "loading training data...";
	for (auto& p : recursive_directory_iterator(dir)) {
		string ext = p.path().extension().string();
		string name = p.path().stem().string();
		if (ext != ".jpg" && ext != ".jpeg"
			&& ext != ".png" && ext != ".pgm")
			continue;
		imgs.push_back(imread(p.path().string()));
		name = name.substr(0, name.rfind('_'));
		if (name2label.find(name) == name2label.end()) {
			labels.push_back(label);
			name2label[name] = label++;
			label2name.push_back(name);
		}
		else {
			labels.push_back(name2label[name]);
		}
	}
	cout << "\t-- ok" << endl;
	cout << "saving faces into face_database.yaml";
	FileStorage fs("face_database.yaml", FileStorage::WRITE);
	fs << "face_db" << "[";
	for (auto &e : imgs) fs << e;
	fs << "]";
	fs.release();
	cout << "\t-- ok" << endl;
	cout << "training...";
	for (auto &e : imgs) {
		preprocess(e, width, height);
	}
	model->train(imgs, labels);
	for (int i = 0; i < label2name.size(); i++)
		model->setLabelInfo(i, label2name[i]);
	cout << "\t-- ok" << endl;
	cout << "+-----------end Train-----------+" << endl;
}
void FaceReg::test(string dir) {
	vector<Mat> image;
	vector<int> labels;
	map<string, int> name2label;
	cout << "+----------Begin  Test----------+" << endl;
	cout << "loading testing data...";
	for (int i = 0; model->getLabelInfo(i) != ""; i++)
		name2label[model->getLabelInfo(i)] = i;
	for (auto& p : recursive_directory_iterator(dir)) {
		string ext = p.path().extension().string();
		string name = p.path().stem().string();
		if (ext != ".jpg" && ext != ".jpeg"
			&& ext != ".png" && ext != ".pgm")
			continue;
		name = name.substr(0, name.rfind('_'));
		image.push_back(imread(p.path().string(), IMREAD_GRAYSCALE));
		labels.push_back(name2label[name]);
	}
	cout << "\t-- ok" << endl;
	cout << "testing...";
	for (auto &e : image)
		preprocess(e, width, height);
	vector<float> cmc(10);
	for (auto &e : cmc) e = 0;
	for (int i = 0; i < image.size(); i++)
	{
		vector<int> index;
		pc->clear();
		model->predict(image[i], pc);
		sort(pc->getDists(), index); 
		int k = 0;
		for (; k < index.size(); k++)
			if (pc->getLabels()[index[k]] == labels[i])
				break;
		for (; k < 10; k++)
			cmc[k]++;
	}
	cout << "\t-- ok" << endl;
	cout << "drawing chart...";
	for (auto&e : cmc) e /= image.size();
	Mat pin;
	drawCMC(cmc, pin);
	cout << "\t-- ok" << endl;
	namedWindow("CMC");
	imshow("CMC", pin);
	cout << "+-----------End  Test-----------+" << endl;
	waitKey();
}
string FaceReg::getMostSimilar(Mat &tar, Mat &sim) {
	Mat e = tar.clone();
	preprocess(e, width, height);
	pc->clear();
	model->predict(e, pc);
	return model->getLabelInfo(pc->getMostSimilar(sim));
}
void FaceReg::load(string path) {
	model->load(path);
}
void FaceReg::save(string path) {
	model->save(path);
}