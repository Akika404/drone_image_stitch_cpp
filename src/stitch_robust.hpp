#pragma once

#include <opencv2/opencv.hpp>

#include <string>
#include <vector>

#include "stitch_config.hpp"

cv::Mat stitchRobustly(
    const std::vector<cv::Mat> &images,
    cv::Stitcher::Mode mode,
    const std::string &stage_name,
    const StitchTuning &tuning,
    int range_width_override = -1,
    const std::vector<std::string> *image_tags = nullptr);
