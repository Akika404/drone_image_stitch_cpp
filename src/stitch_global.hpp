#pragma once

#include <opencv2/opencv.hpp>

#include <vector>

#include "pos_image_grouper.hpp"
#include "stitch_config.hpp"

struct GlobalStitchInput {
    std::vector<cv::Mat> images;
    cv::UMat match_mask;
    int num_within_pairs = 0;
    int num_cross_pairs = 0;
};

void orderStripGroupsByCrossTrack(std::vector<FlightStripGroup> &groups);

GlobalStitchInput buildGlobalStitchInput(
    const std::vector<FlightStripGroup> &strip_groups,
    const StitchTuning &tuning);

cv::Mat stitchGlobalPipeline(
    const std::vector<cv::Mat> &images,
    const cv::UMat &match_mask,
    const StitchTuning &tuning);
