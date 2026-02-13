#pragma once

struct StitchTuning {
    int sift_features = 1500;
    float match_conf = 0.35f;
    int min_good_matches = 10;
    int min_inliers = 8;

    bool use_range_matcher = true;
    int range_width = 6;

    bool use_affine_bundle = true;
    bool use_affine_warper = true;
    bool use_blocks_gain = true;
    int blend_bands = 5;
    float pano_conf_thresh = 0.7f;

    bool use_anchor_fallback = true;
    int anchor_window = 4;

    bool use_opencl = true;
    bool try_gpu = true;

    double registration_resol_mpx = 0.40;
    double seam_estimation_resol_mpx = 0.10;
    double compositing_resol_mpx = -1.0;
    bool adaptive_speed = true;
    int large_strip_threshold = 36;
    int large_strip_sift_features = 1000;
    int large_strip_range_width = 4;
};

StitchTuning loadStitchTuning();
