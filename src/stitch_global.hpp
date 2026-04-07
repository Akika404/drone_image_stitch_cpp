#pragma once

#include <opencv2/opencv.hpp>

#include <vector>

#include "stitch_config.hpp"

/// 自定义航带间拼接（不使用 cv::Stitcher）：
/// 按相邻航带顺序估计仿射变换并做多频段融合。
cv::Mat stitchInterStripsCustom(
    const std::vector<cv::Mat> &strip_panoramas,
    const StitchTuning &tuning);
