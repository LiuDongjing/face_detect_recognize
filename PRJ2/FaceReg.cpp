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
	clog << "[��ʼѵ��]" << endl << endl;
	clog << "����ѵ������..." << endl << endl;
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
	clog << "ѵ������������ϣ�" << endl << endl;
	clog << "���������ݱ��浽 " << facePath << endl << endl;
	FileStorage fs(facePath, FileStorage::WRITE);
	fs << "face_db" << "[";
	for (auto &e : imgs) {
		Mat t = e.clone();
		resize(t, t, Size(), 0.3, 0.3); //�����ͼ������ΪСͼ��չʾ�ģ����ð���ֱ������洢
		fs << t;
	}
	fs << "]";
	fs.release();
	clog << "���������ѱ����� " << facePath << endl << endl;
	clog << "**" << imgs.size() << "��ѵ������, " << label2name.size() << "�����**" << endl << endl;
	clock_t st = clock();
	clog << "ѵ����..." << endl << endl;
	for (auto &e : imgs) {
		preprocess(e, width, height);
	}
	model->train(imgs, labels);
	for (int i = 0; i < label2name.size(); i++)
		model->setLabelInfo(i, label2name[i]);
	clog << "ѵ����ʱ" << 1000.0 * (clock() - st) / CLOCKS_PER_SEC << "����." << endl << endl;
	clog << "[ѵ������]" << endl << endl;
}
void FaceReg::test(string dir, string facePath) {
	vector<Mat> image;
	vector<int> labels;
	map<string, int> name2label;
	clog << "[��ʼ����]" << endl << endl;
	clog << "�����������..." << endl << endl;
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
	clog << "��������������ϣ�" << endl << endl;
	clog << "**" << image.size() << "����������, " << name2label.size() << "�����**" << endl << endl;
	clog << "������..." << endl << endl;
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
	clog << "���Ժ�ʱ" << 1000.0 * (clock() - st) / CLOCKS_PER_SEC << "����." << endl << endl;
	clog << "[���Խ���]" << endl << endl;
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