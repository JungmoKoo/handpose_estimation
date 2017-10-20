#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>

#include <caffe/caffe.hpp>

#include <iostream>

#include "fcn/fcn_utils.h"

using namespace caffe;  // NOLINT(build/namespaces)
using std::string;
using namespace std;
using namespace cv;


string labels[8] = {"Background", "Head", "Left shoulder", "Left elbow", "Left hand", "Right shoulder", "Right elbow", "Right hand"};
cv::Vec3b colorMap[8] = {
  cv::Vec3b(0, 0, 0),
  cv::Vec3b(0, 0, 0),
  cv::Vec3b(0, 0, 0),
  cv::Vec3b(0, 0, 0),
  cv::Vec3b(1, 1, 1),
  cv::Vec3b(0, 0, 0),
  cv::Vec3b(0, 0, 0),
  cv::Vec3b(1, 1, 1) };

// please edit path
string model_file   = "/home/jungmo/catkin_ws/src/handpose_estimation/models/deploy.prototxt";
string trained_file = "/home/jungmo/catkin_ws/src/handpose_estimation/models/snapshot.caffemodel";

FCNUtils FCNUtils(model_file, trained_file);

cv::Mat img;

void imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
  try
  {
    cv_bridge::toCvShare(msg, "bgr8")->image.copyTo(img);

    if (img.empty())
    {
      cout << "img empty" << endl;
    }

    FCNUtils.Inference(img);
    if ( !FCNUtils.net_->has_blob("score"))
      cout << "invalid layer" << endl;
    const boost::shared_ptr<Blob<float> > output_layer_in = FCNUtils.net_->blob_by_name("score");                    // assumes there exists a 'prob' blob
    const float* output_data = output_layer_in->cpu_data();

    cv::Mat maxProbMap(cv::Size(output_layer_in->width(), output_layer_in->height()), CV_32FC1);
    maxProbMap.setTo(-9999);
    cv::Mat classColorMap(cv::Size(output_layer_in->width(), output_layer_in->height()), CV_8UC3);

    float prob;
    for (int n = 0; n < output_layer_in->channels(); n++)
    {
      int ch_offset = n*(output_layer_in->width()*output_layer_in->height());
      for (int y = 0; y < output_layer_in->height(); y++)
      {
        for ( int x = 0; x < output_layer_in->width(); x++)
        {
          prob = output_data[ch_offset + y*output_layer_in->width() + x];
          if (prob > maxProbMap.at<float>(cv::Point(x, y)))
          {
            maxProbMap.at<float>(cv::Point(x, y)) = prob;
            classColorMap.at<Vec3b>(cv::Point(x, y)) = colorMap[n];
          }
        }
      }
    }

    cv::Mat output;
    output = img.mul(classColorMap);

    cv::imshow("output", output);
    cv::waitKey(1);
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
  }
}


int main(int argc, char** argv)
{
  ros::init(argc, argv, "handpose_estimation");
  ros::NodeHandle nh;

  image_transport::ImageTransport it(nh);
  image_transport::Subscriber sub = it.subscribe("camera/rgb/image_raw", 1, imageCallback);

  while(1)
  {
    ros::spinOnce();
  }

  return 0;
}
