/*ͳһ����ѵ�������Ժ�ʶ����̵��ࡣÿ������ʶ���㷨���̳���FaceRecognizer��
  FaceRegͨ�������FaceRecognizer���󣬵�������train��predict��������ɲ�ͬ
  �µ�ѵ�������Ժ�ʶ�����*/
#pragma once
#include <opencv2/core/core.hpp>
#include <opencv2/face/facerec.hpp>
#include "Collector.h"
using namespace std;
using namespace cv;
using namespace face;
class FaceReg {
public:
	/*fr��ʶ���㷨ʵ����w��h�ֱ���Ԥ�����ͼ��ĳ��Ϳ�facePath��
	���������л��󱣴�·��*/
	FaceReg(Ptr<FaceRecognizer> fr, int w, int h);
	FaceReg(Ptr<FaceRecognizer> fr, int w, int h, string facePath);

	/*ѵ��ģ�ͣ�dir������ѵ����ԭʼͼ�񼯣�facePathͬ��*/
	void train(string dir, string facePath = "face_database.yaml");

	/*����ģ�ͣ�dir�����˲��Ե�ԭʼͼ�񼯣�facePathͬ��*/
	void test(string dir, string facePath = "face_database.yaml");

	/*��ȡ������tar���������sim�������������ƾ�����ֵ�õ���facePathͬ��*/
	string getMostSimilar(Mat &tar, Mat &sim, string facePath = "face_database.yaml");

	/*����ģ��*/
	void load(string path);

	/*����ģ��*/
	void save(string path);
private:
	Ptr<FaceRecognizer> model;
	Ptr<Collector> pc;
	int width;
	int height;
};

