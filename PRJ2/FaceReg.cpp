#include "FaceReg.h"
#include "toolkits.h"
#include "Collector.h"
#include <set>
#include <vector>
#include <experimental/filesystem>
#include <iostream>
#include <time.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
namespace fs = std::experimental::filesystem;
using namespace fs;
FaceReg::FaceReg(Ptr<FaceRecognizer> fr, int w, int h) {
	model = fr;
	width = w;
	height = h;
}
FaceReg::FaceReg(Ptr<FaceRecognizer> fr, int w, int h, string facePath) {
	model = fr;
	width = w;
	height = h;
	pc = new Collector(facePath);
}
void FaceReg::train(string dir, string facePath) {
	vector<string> label2name;
	set<string> nameGot;
	vector<Mat> imgs;
	map<string, int> name2label;
	vector<int> labels;
	int label = 0;
	clog << "[开始训练]" << endl << endl;
	clog << "载入训练数据..." << endl << endl;
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
	clog << "训练数据载入完毕！" << endl << endl;
	clog << "将人脸数据保存到 " << facePath << endl << endl;
	FileStorage fs(facePath, FileStorage::WRITE);
	fs << "face_db" << "[";
	for (auto &e : imgs) {
		Mat t = e.clone();
		resize(t, t, Size(), 0.3, 0.3); //这里的图像是作为小图像展示的，不用按大分辨率来存储
		fs << t;
	}
	fs << "]";
	fs.release();
	clog << "人脸数据已保存至 " << facePath << endl << endl;
	clog << "**" << imgs.size() << "个训练样本, " << label2name.size() << "个类别**" << endl << endl;
	clock_t st = clock();
	clog << "训练中..." << endl << endl;
	for (auto &e : imgs) {
		preprocess(e, width, height);
	}
	model->train(imgs, labels);
	for (int i = 0; i < label2name.size(); i++)
		model->setLabelInfo(i, label2name[i]);
	clog << "训练耗时" << 1000.0 * (clock() - st) / CLOCKS_PER_SEC << "毫秒." << endl << endl;
	clog << "[训练结束]" << endl << endl;
}
void FaceReg::test(string dir, string facePath) {
	vector<Mat> image;
	vector<int> labels;
	map<string, int> name2label;
	clog << "[开始测试]" << endl << endl;
	clog << "载入测试数据..." << endl << endl;
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
	clog << "测试数据载入完毕！" << endl << endl;
	clog << "**" << image.size() << "个测试样本, " << name2label.size() << "个类别**" << endl << endl;
	clog << "测试中..." << endl << endl;
	clock_t st = clock();
	for (auto &e : image)
		preprocess(e, width, height);
	vector<float> cmc(10);
	for (auto &e : cmc) e = 0;
	if(pc.empty())
		pc = new Collector(facePath);
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
	for (auto&e : cmc) e /= image.size();
	Mat pin;
	drawCMC(cmc, pin);
	namedWindow("CMC");
	imshow("CMC", pin);
	clog << "测试耗时" << 1000.0 * (clock() - st) / CLOCKS_PER_SEC << "毫秒." << endl << endl;
	clog << "[测试结束]" << endl << endl;
	waitKey();
}
string FaceReg::getMostSimilar(Mat &tar, Mat &sim, string facePath) {
	Mat e = tar.clone();
	preprocess(e, width, height);
	if (pc.empty())
		pc = new Collector(facePath);
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