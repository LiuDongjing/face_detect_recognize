#pragma once
#include <iostream>
#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
using namespace std;
using namespace cv;
bool pretrain(string conf, string dir);

bool loadTrainData(string conf, vector<Mat> &image, vector<int> &label);
bool configure(string conf, vector<string> &label2name,
	vector<string> &sample2path);
void mark(Mat &panel, Rect reg, string title, Mat &mini, bool draw = true);
void preprocess(Mat &img, int w, int h, bool cvt = false);
void train(CommandLineParser &parser, bool test = true);
void sort(vector<double> &data, vector<int>& index, bool asc = true);
void drawCMC(vector<float> &data, Mat &img);
void centerText(Mat &img, string info, Point o, double scale = 1, Scalar color = Scalar(0, 0, 255), int font = FONT_HERSHEY_SIMPLEX);