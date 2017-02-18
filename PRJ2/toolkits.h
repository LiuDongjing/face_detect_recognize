#pragma once
#include <iostream>
#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
using namespace std;
using namespace cv;
void mark(Mat &panel, Rect reg, string title, Mat &mini, bool draw = true);
void preprocess(Mat &img, int w, int h, bool cvt = false);
void sort(vector<double> &data, vector<int>& index, bool asc = true);
void drawCMC(vector<float> &data, Mat &img);