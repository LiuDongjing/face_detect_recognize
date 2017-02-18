#pragma once
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/face/facerec.hpp>
#include <iostream>
// to-do: �ɱ�뾶�ھӣ��淶����LBP�����ֲ�ͬ��cell��knn���ࣻ
using namespace std;
using namespace cv;
using namespace face;
class MyLBPHFace : public FaceRecognizer {
public :
	MyLBPHFace();
	MyLBPHFace(int x, int y);
	double getThreshold() const override;
	void load(const FileStorage &fs) override;
	void predict(InputArray src, Ptr<PredictCollector> collector) const override;
	void save(FileStorage &fs) const override;
	void train(InputArrayOfArrays src, InputArray labels) override;
	void setThreshold(double thresh) override;
	String getLabelInfo(int label) const  override;
	void setLabelInfo(int label, const String &strInfo) override;
private:
	void getDescVec(Mat &img, Mat &vec) const;
	void encode(Mat &img) const;
	void hist(Mat &img, Mat &h) const;
private:
	Mat desc;//NxM�ľ���N��������������M��LBPH�������Ĵ�С
	vector<int> lab;//N��������Ӧ�ı�ǩ
	int nX = 5; //��һ��ͼ��ˮƽ�����Ϸֳ�nX��
	int nY = 5; //��ͼ��ˮƽ�ֳ�nY��
	double radius = 1; //�ھӰ뾶��Ŀǰδʹ��
	int num = 8;// �ھӵ�������Ŀǰδʹ��
	double threshold;
	vector<string> labInfo;
};
Ptr<MyLBPHFace> createMyLBPHFaceRecognizer();
Ptr<MyLBPHFace> createMyLBPHFaceRecognizer(int x, int y);