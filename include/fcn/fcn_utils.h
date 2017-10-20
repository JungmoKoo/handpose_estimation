#include <caffe/caffe.hpp>

#include <opencv2/opencv.hpp>

//#endif  // USE_OPENCV
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

// jungmo
#include <fstream>
#include <iostream>

//#ifdef USE_OPENCV

using namespace caffe;  // NOLINT(build/namespaces)
using std::string;
using namespace std;
using namespace cv;

class FCNUtils
{
public:
    FCNUtils(const string& model_file, const string& trained_file);
    Blob<float>* Inference(const cv::Mat& img);

    void WrapInputLayer(std::vector<cv::Mat>* input_channels);
    void Preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels);

    boost::shared_ptr<Net<float> > net_;

private:
    //void WrapInputLayer(std::vector<cv::Mat>* input_channels);
    //void Preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels);

private:
//    boost::shared_ptr<Net<float> > net_;
    cv::Size input_geometry_;
    int num_channels_;
    cv::Mat mean_;
    std::vector<string> labels_;
};
