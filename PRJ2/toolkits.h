/*һЩͨ�õĹ��ߺ���*/
#pragma once
#include <iostream>
#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
using namespace std;
using namespace cv;
/*��panel�ﻭ��һ�����ο�reg������������ʾtitle��mini��С����panel
  �����Ͻ���ʾ(draw == true��ʾ��������ʾ)*/
void mark(Mat &panel, Rect reg, string title, Mat &mini, bool draw = true);

/*��ͼ�����w x h�Ҷ�ͼ*/
void preprocess(Mat &img, int w, int h);

/*��data����Ĭ�ϴ�С����(asc = true)��index����������ÿ��Ԫ����data�е�ԭʼλ��*/
void sort(vector<double> &data, vector<int>& index, bool asc = true);

/*��img�ϻ���cmc@10����ͼ*/
void drawCMC(vector<float> &data, Mat &img);