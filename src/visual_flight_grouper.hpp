#pragma once

#include <opencv2/opencv.hpp>

#include <string>
#include <vector>

#include "stitch_config.hpp"

struct VisualStripGroup {
    std::vector<cv::Mat> images;
    std::vector<std::string> image_ids;
};

class VisualFlightGrouper {
public:
    [[nodiscard]] static std::vector<VisualStripGroup> groupBoustrophedon(
        const std::vector<cv::Mat> &images,
        const std::vector<std::string> &image_ids,
        const StitchTuning &tuning);
};
