#pragma once

#include <opencv2/opencv.hpp>

#include <string>
#include <vector>

struct LoadedImages {
    std::vector<cv::Mat> images;
    std::vector<std::string> ids;
    std::vector<std::string> paths;
};

class ImageLoader {
public:
    static std::vector<cv::Mat> load(const std::string &folder);

    static LoadedImages loadWithIds(const std::string &folder);

    static LoadedImages listWithIds(const std::string &folder);
};
