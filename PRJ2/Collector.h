/*
	�̳���PredictCollector����д��collect������
 �ڵ���ģ�͵�predict(InputArray, PredictCollector)ʱ���ռ�Ԥ������е����ݡ�
	eigenface��fisherface��lbphface��predict(InputArray, PredictCollector)������
 ������(˳���train�������������vectorһ��)��������������������������������
 ��������������������֮��ľ���dist��Ȼ���������Ӧ��label��dist����PredictCollector
 ��collect������
	���PredictCollector���Խ��������������ÿ�������ľ����Ѿ�����
 ������label�����������������Collector����������ô���ġ������������������ݣ���
 ������kNN���������ࡣ
*/
#pragma once
#include <opencv2/core/core.hpp>
#include <opencv2/face/facerec.hpp>
using namespace std; 
using namespace cv;
using namespace face;
class Collector :public PredictCollector {
public:
	/* �������ݿ�·������ΪgetMostSimilar����Ҫ������
	   �����������������������ͼ��*/
	Collector(string facedatabase);

	/*��ģ�͵�predict�������ã�ע�������label������Ԥ��õ���label������
	  ����������label;�÷�����Ѵ�Ԥ��������ÿ�����������ľ��붼��������*/
	bool collect(int label, double dist) override;
	void init(size_t size) override;

	/*���ر���Ԥ�������Ԥ�����������������������ͼ��ͨ��e���أ�������label
	  ͨ������ֵ�õ�*/
	int getMostSimilar(Mat &e);

	/*���ش˴�Ԥ��collect�õ��ľ�������*/
	vector<double> &getDists();

	/*���ؿ���������label*/
	vector<int> &getLabels();

	/*����ѱ�������ݣ�ע��ֻ�����dists���飬��Ϊ����������ÿ��Ԥ�ⶼ��һ����*/
	void clear();
private:
	vector<int> labels;
	vector<double> dists;
	vector<Mat> facedb;
};
