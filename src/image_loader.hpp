#pragma once

#include <opencv2/opencv.hpp>

#include <string>
#include <vector>

class ImageLoader {
public:
    static std::vector<cv::Mat> load(const std::string &folder);
};
