#pragma once

#include <opencv2/opencv.hpp>

void autoCropBlackBorder(cv::Mat &pano);

std::string stitchStatusToString(cv::Stitcher::Status status);
