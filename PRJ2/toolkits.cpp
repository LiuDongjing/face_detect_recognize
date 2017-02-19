#include "toolkits.h"
#include <fstream>
#include <experimental/filesystem>
#include <map>
#include <opencv2/face/facerec.hpp>
#include <algorithm>
namespace fs = std::experimental::filesystem;
using namespace fs;
using namespace face;

//文本居中显示
void centerText(Mat &img, string info, Point o, double scale = 1, 
	Scalar color = Scalar(0, 0, 255), int font = FONT_HERSHEY_SIMPLEX);

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
void preprocess(Mat &img, int w, int h) {
	if(img.channels() > 1)
		cvtColor(img, img, CV_BGR2GRAY);
	if(img.cols != w || img.rows != h)
		resize(img, img, Size(w, h));
}
void sort(vector<double> &data, vector<int>& index, bool asc)
{
	//采用的是冒泡排序算法
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
	arrowedLine(win, Point(BW, BW + SZ - 1 + 10), Point(BW, 10), Scalar(255, 255, 255), 1, 8, 0, 0.02);
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