/*一些通用的工具函数*/
#pragma once
#include <iostream>
#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
using namespace std;
using namespace cv;
/*在panel里画出一个矩形框reg，并在其上显示title；mini缩小后再panel
  的左上角显示(draw == true显示，否则不显示)*/
void mark(Mat &panel, Rect reg, string title, Mat &mini, bool draw = true);

/*将图像处理成w x h灰度图*/
void preprocess(Mat &img, int w, int h);

/*给data排序，默认从小到大(asc = true)；index返回排序后的每个元素在data中的原始位置*/
void sort(vector<double> &data, vector<int>& index, bool asc = true);

/*在img上绘制cmc@10曲线图*/
void drawCMC(vector<float> &data, Mat &img);