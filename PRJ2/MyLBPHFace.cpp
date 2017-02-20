#include "MyLBPHFace.h"
#include <cmath>
MyLBPHFace::MyLBPHFace() {

}
MyLBPHFace::MyLBPHFace(int x, int y) {
	assert(x > 0 && y > 0);
	nX = x;
	nY = y;
}
double MyLBPHFace::getThreshold() const {
	return threshold;
}
void MyLBPHFace::load(const FileStorage &fs) {
	fs["nX"] >> nX;
	fs["nY"] >> nY;
	fs["radius"] >> radius;
	fs["num"] >> num;
	fs["threshold"] >> threshold;
	fs["desc"] >> desc;
	for (auto &e : fs["lab"]) {
		int k;
		e >> k;
		lab.push_back(k);
	}
	for (auto &e : fs["labInfo"]) {
		string s;
		e >> s;
		labInfo.push_back(s);
	}
}

void MyLBPHFace::predict(InputArray src, Ptr<PredictCollector> collector) const{
	Mat img;
	src.copyTo(img);//避免修改原来的图像
	Mat vec(1, nX*nY * 256, CV_32FC1, Scalar(0));//分配好特征向量的内存并全部初始化为0
	getDescVec(img, vec);//获取待预测人脸的特征向量
	for (int i = 0; i < lab.size(); i++) {
		//遍历库中每张人脸的特征向量，每个特征向量占一行
		Mat t = desc(Rect(0, i, nX*nY * 256, 1));
		
		//传入每张人脸的label及对应的欧式距离
		collector->collect(lab[i], norm(vec, t));
	}
}
void MyLBPHFace::save(FileStorage &fs) const{
	fs << "nX" << nX;
	fs << "nY" << nY;
	fs << "radius" << radius;
	fs << "num" << num;
	fs << "threshold" << threshold;
	fs << "desc" << desc;
	fs << "lab" << "[";
	for (auto &e : lab) fs << e;
	fs << "]";
	fs << "labInfo" << "[";
	for (auto &e : labInfo) fs << e;
	fs << "]";
}
void MyLBPHFace::train(InputArrayOfArrays src, InputArray labels) {
	vector<Mat> imgs;

	/*将labels保存到lab中并未labInfo分配内存
	  注意lab是每张人脸的标签，而labInfo是每
	  个标签的信息，两者的大小并不一致*/
	{
		//vector<int> 转成Mat是1xN，vs显示的时候是1xNx1，第一个是通道数
		Mat tmp = labels.getMat();
		int ml = 0;
		for (int i = 0; i < tmp.total(); i++)
		{
			int t = tmp.at<int>(0, i);
			lab.push_back(t);
			if (t > ml) ml = t;
		}
		labInfo = vector<string>(ml);
	}
	src.getMatVector(imgs);
	assert(imgs.size() > 0);
	assert(imgs.size() == labels.total());
	int row = imgs[0].rows, col = imgs[0].cols;

	//每个样本图片的特征向量占一行
	//分配内存并全部初始化为0
	desc = Mat(imgs.size(), nX*nY*256, CV_32FC1, Scalar(0));
	for (int i = 0; i < imgs.size(); i++) {
		assert(imgs[i].rows == row && imgs[i].cols == col);
		
		//计算库中人脸的特征向量
		getDescVec(imgs[i], desc(Rect(0, i, nX*nY*256, 1)));
	}
}
void MyLBPHFace::setThreshold(double thresh) {
	threshold = thresh;
}
String MyLBPHFace::getLabelInfo(int label) const {
	if(label < labInfo.size())
		return labInfo[label];
	return "";
}
void MyLBPHFace::setLabelInfo(int label, const String &strInfo) {
	if (label < labInfo.size())
		labInfo[label] = strInfo;
}
void MyLBPHFace::getDescVec(Mat &img, Mat &vec) const{
	//vec的内存已经分配好了
	int ix = ceil(img.cols / float(nX));
	int iy = ceil(img.rows / float(nY));
	encode(img);
	int offset = 0;
	for (int i = 0; i < img.cols; i += ix) {
		for (int j = 0; j < img.rows; j += iy) {
			int w = (i + ix) < img.cols ? ix : img.cols - i;
			int h = (j + iy) < img.rows ? iy : img.rows - j;
			//分别计算每个图像区域的直方图
			hist(img(Rect(i, j, w, h)), vec(Rect(offset, 0, 256, 1)));
			offset += 256;
		}
	}
}
void MyLBPHFace::encode(Mat &img) const{
/* 邻居的顺序如下
  701
  6c2
  543*/
	int xs[8] = { 0, 1, 1, 1, 0, -1, -1, -1 };
	int ys[8] = { -1, -1, 0, 1, 1, 1, 0, -1 };
	Mat tmp = img.clone();
	for (int i = 0; i < tmp.rows; i++) {
		for (int j = 0; j < tmp.cols; j++) {
			uint8_t c = 0;
			for (int z = 0; z < 8; z++) {
				int x = i + xs[z];
				int y = j + ys[z];
				//超出图像区域视为0
				if (x < 0 || x >= tmp.rows || y < 0 || y >= tmp.cols) continue;
				if (tmp.at<uint8_t>(x, y) >= tmp.at<uint8_t>(i, j))
					c += 1 << z;
			}
			img.at<uint8_t>(i, j) = c;
		}
	}
}
void MyLBPHFace::hist(Mat &img, Mat &h) const{
	for(int i = 0;i < img.rows;i++)
		for (int j = 0; j < img.cols; j++) {
			h.at<float>(0, img.at<uint8_t>(i, j))++;
		}
}
Ptr<MyLBPHFace> createMyLBPHFaceRecognizer() {
	return new MyLBPHFace();
}
Ptr<MyLBPHFace> createMyLBPHFaceRecognizer(int x, int y) {
	return new MyLBPHFace(x, y);
}