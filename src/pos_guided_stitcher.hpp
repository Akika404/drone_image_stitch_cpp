#pragma once

#include <string>
#include <vector>

#include "pos_record.hpp"
#include "stitch_config.hpp"

struct StitchFrame {
    std::string id;
    std::string path;
    PosRecord pos;
};

struct StitchOutput {
    bool ok = false;
    std::string message;
    std::string output_path;
    int input_count = 0;
    int stitched_count = 0;
};

StitchOutput stitchWithPosGuidance(const std::vector<StitchFrame> &frames,
                                   const StitchTuning &tuning,
                                   const std::string &output_path);
