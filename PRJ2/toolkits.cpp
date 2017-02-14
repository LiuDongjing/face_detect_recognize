#include "toolkits.h"
#include <fstream>
#include <experimental/filesystem>
#include <map>
#include <opencv2/face/facerec.hpp>
#include <algorithm>
#include "EigenFace.h"
#include "FaceRegBase.h"
namespace fs = std::experimental::filesystem;
using namespace fs;
using namespace face;
bool pretrain(string conf, string dir) {
	int label = 0;
	map<string, int> tab;
	vector<string> label2name;
	vector<string> sample2path;
	map<string, int> name2label;
	for (auto& p : recursive_directory_iterator(dir)) {
		string ext = p.path().extension().string();
		string name = p.path().stem().string();
		if (ext != ".jpg" && ext != ".jpeg"
			&& ext != ".png" && ext != ".pgm")
			continue;
		name = name.substr(0, name.rfind('_'));
		if (name2label.find(name) != name2label.end())
		{
			tab[p.path().string()] = name2label[name];
			sample2path.push_back(p.path().string());
		}
		else {
			tab[p.path().string()] = label;
			label2name.push_back(name);
			sample2path.push_back(p.path().string());
			name2label[name] = label;
			label++;
		}
	}
	FileStorage fs(conf, FileStorage::WRITE);
	if (!fs.isOpened()) {
		cerr << "Configure failed!" << endl;
		return false;
	}
	fs << "path" << "[";
	for (auto &e : tab) {
		fs << e.first;
	}
	fs << "]";
	fs << "label" << "[";
	for (auto &e : tab) {
		fs << e.second;
	}
	fs << "]";
	fs << "label_name" << "[";
	for (auto &e : label2name) {
		fs << e;
	}
	fs << "]";

	fs << "sample_path" << "[";
	for (auto &e : sample2path) {
		fs << e;
	}
	fs << "]";
	return true;
}
bool loadTestData(string path, vector<Mat> &image, vector<string> &label) {
	for (auto& p : recursive_directory_iterator(path)) {
		string ext = p.path().extension().string();
		string name = p.path().stem().string();
		if (ext != ".jpg" && ext != ".jpeg"
			&& ext != ".png" && ext != ".pgm")
			continue;
		name = name.substr(0, name.rfind('_'));
		image.push_back(imread(p.path().string(), IMREAD_GRAYSCALE));
		label.push_back(name);
	}
	return true;
}
bool loadTrainData(string conf, vector<Mat> &image, vector<int> &label) {
	FileStorage fs(conf, FileStorage::READ);
	if (!fs.isOpened()) {
		cerr << "Open configure file failed!" << endl;
		return false;
	}
	FileNode pathNode = fs["path"];
	if (pathNode.type() != FileNode::SEQ) {
		cerr << "Error in configure file!" << endl;
		return false;
	}
	for (auto &e : pathNode)
		image.push_back(imread((string)e, IMREAD_GRAYSCALE));

	FileNode labelNode = fs["label"];
	if (labelNode.type() != FileNode::SEQ) {
		cerr << "Error in configure file!" << endl;
		return false;
	}
	for (auto &e : labelNode)
		label.push_back((int)e);
	return true;
}
bool configure(string conf, vector<string> &label2name,
	vector<string> &sample2path) {
	FileStorage fs(conf, FileStorage::READ);
	FileNode lnNode = fs["label_name"];
	if (lnNode.type() != FileNode::SEQ) {
		cerr << "Error in configure file!" << endl;
		return false;
	}
	for (auto &e : lnNode)
		label2name.push_back((string)e);

	FileNode lpNode = fs["sample_path"];
	if (lpNode.type() != FileNode::SEQ) {
		cerr << "Error in configure file!" << endl;
		return false;
	}
	for (auto &e : lpNode)
		sample2path.push_back((string)e);
	return true;
}
void mark(Mat &panel, Rect reg, string title, Mat &mini, bool draw) {
	rectangle(panel, Point(reg.x, reg.y), Point(reg.x + reg.width,
		reg.y + reg.height), Scalar(0, 0, 255));
	int fontFace = FONT_HERSHEY_SIMPLEX;
	double fontScale = 1;
	int thickness = 1;
	int baseline = 0;
	Size textSize = getTextSize(title, fontFace,
		fontScale, thickness, &baseline);
	Point org(reg.x + (reg.width - textSize.width) / 2, 
		reg.y + textSize.height);
	putText(panel, title, org, fontFace, fontScale, Scalar(0, 238, 238), thickness);
	if (draw) {
		double scale = (panel.cols * 0.2) / mini.cols;
		Mat scaleImg;
		resize(mini, scaleImg, Size(), scale, scale);
		Rect roi(1, 1, scaleImg.cols, scaleImg.rows);
		Mat win = panel(roi);
		scaleImg.copyTo(win);
		rectangle(panel, Point(0, 0), Point(roi.width,
			roi.height), Scalar(0, 0, 255));
	}
}
void preprocess(Mat &img, int w, int h, bool cvt) {
	if(img.channels() > 1)
		cvtColor(img, img, CV_BGR2GRAY);
	if(img.cols != w || img.rows != h)
		resize(img, img, Size(w, h));
	//GaussianBlur(img, img, Size(3, 3), 0, 0, BORDER_DEFAULT);
	if (cvt) {
		img = img.reshape(1, 1);
		img.convertTo(img, CV_64FC1);
	}
}
void train(CommandLineParser &parser, bool test) {
	string traindir = parser.get<string>("train_dir");
	string testdir = parser.get<string>("test_dir");
	string conf = parser.get<string>("conf");
	string modelPath = parser.get<string>("model");
	string method = parser.get<string>("method");
	transform(method.begin(), method.end(), method.begin(), tolower);
	int w = parser.get<int>("w");
	int h = parser.get<int>("h");
	int num = parser.get<int>("num");
	double threshold = parser.get<double>("threshold");
	if (!parser.check()) {
		parser.printErrors();
		return;
	}
	cout << "Preparing training data...";
	pretrain(conf, traindir);
	vector<Mat> train;
	vector<int> label;
	loadTrainData(conf, train, label);
	for (auto &e : train)
		preprocess(e, w, h);
	cout << "    Done" << endl;
	FaceRegBase *model = nullptr;
	cout << "Training...";
	if (method == "eigenface") {
		model = new EigenFace();
		model->train(train, label);
		
		cout << "    Done" << endl << endl;
		if (test) {
			vector<Mat> testData;
			vector<string> names;
			vector<string> label2name;
			vector<string> sample2path;
			cout << "Preparing testing data...";
			if (!configure(conf, label2name, sample2path)) {
				cout << "Configure failed!" << endl;
				return;
			}
			loadTestData(testdir, testData, names);
			cout << "    Done" << endl;
			cout << "Testing..." << endl;
			int cnt = 0;
			double minmax = 0;
			vector<float> cmc(10);
			for (auto &e : cmc) e = 0;
			for (int i = 0; i < testData.size(); i++)
			{
				double dist = 0;
				vector<int> index;
				int sampleNum;
				model->predict(testData[i], sampleNum, dist, index);
				minmax = max(minmax, dist);
				int k = 0;
				for (; k < index.size(); k++)
					if (label2name[label[index[k]]] == names[i])
						break;
				for (; k < 10; k++)
					cmc[k]++;
			}
			model->threshold(minmax * 1.2);
			model->save(modelPath);
			for (auto&e : cmc) e /= testData.size();
			Mat pin;
			drawCMC(cmc, pin);
			namedWindow("CMC");
			imshow("CMC", pin);
			waitKey();
		}
		else if (method == "lbph") {
			//model = createLBPHFaceRecognizer();
		}
		else if (method == "fisher") {
			//暂时不能用，会train时会报错，未经处理的异常
			//model = createFisherFaceRecognizer();
		}
	}
}
void sort(vector<double> &data, vector<int>& index, bool asc)
{
	index.clear();
	for (int i = 0; i < data.size(); i++) index.push_back(i);
	int ext;
	for (int i = 0; i < data.size(); i++) {
		ext = i;
		for (int j = i + 1; j < data.size(); j++) {
			if (data[j] < data[ext] && asc || data[j] > data[ext] && !asc)
				ext = j;
		}
		swap(data[i], data[ext]);
		swap(index[i], index[ext]);
	}
}
void drawCMC(vector<float> &data, Mat &img) {
#define SZ 600
#define BW 50
	Mat win(SZ + BW * 2, SZ + BW * 2, CV_8UC3, Scalar(0, 0, 0));
	int d = (SZ-1) / 10;
	float by = data[0];
	float dy = (SZ-1) / (1 - by);
	Point p0(BW, BW + SZ - 1);
	arrowedLine(win, Point(BW - 10, BW + SZ - 1), Point(SZ + 2*BW - 10, BW + SZ - 1), Scalar(255, 255, 255), 1, 8, 0, 0.02);
	//line(win, Point(SZ + 10 - 20, BW + SZ - 1 + 10), Point(SZ + 10, BW + SZ - 1), Scalar(255, 0, 0));
	
	arrowedLine(win, Point(BW, BW + SZ - 1 + 10), Point(BW, 10), Scalar(255, 255, 255), 1, 8, 0, 0.02);
	//line(win, Point(BW, BW - 10), Point(BW - 10, BW - 10 + 20), Scalar(255, 0, 0));
	for (int i = 1; i <= 10; i++) {
		Point p1(BW + i*d, BW + round(SZ -1 - (data[i - 1] - by) * dy));
		circle(win, p1, 5, Scalar(0, 255, 0));
		string info = format("%0.2f", (data[i - 1]));
		centerText(win, info, p1, 0.5, Scalar(0, 0, 255));
		line(win, p0, p1, Scalar(0, 255, 255));
		p0 = p1;
	}
	img = win.clone();
}
void centerText(Mat &img, string info, Point o, double scale, Scalar color, int font) {
	Size fs = getTextSize(info, font, scale, 1, nullptr);
	putText(img, info, o - Point(fs.width / 2, 5), font, scale, color);
}