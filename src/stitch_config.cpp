#include "stitch_config.hpp"

#include <cstdlib>
#include <string>

namespace {
    int envIntOr(const char *name, const int default_value, const int min_value) {
        const char *value = std::getenv(name);
        if (!value) {
            return default_value;
        }
        char *end = nullptr;
        const long parsed = std::strtol(value, &end, 10);
        if (end == value || *end != '\0') {
            return default_value;
        }
        if (parsed < min_value) {
            return min_value;
        }
        return static_cast<int>(parsed);
    }

    float envFloatOr(const char *name, const float default_value, const float min_value) {
        const char *value = std::getenv(name);
        if (!value) {
            return default_value;
        }
        char *end = nullptr;
        const float parsed = std::strtof(value, &end);
        if (end == value || *end != '\0') {
            return default_value;
        }
        if (parsed < min_value) {
            return min_value;
        }
        return parsed;
    }
} // namespace

bool envEnabled(const char *name, const bool default_value) {
    const char *value = std::getenv(name);
    if (!value) {
        return default_value;
    }
    const std::string text(value);
    if (text == "1" || text == "true" || text == "TRUE" || text == "on" || text == "ON") {
        return true;
    }
    if (text == "0" || text == "false" || text == "FALSE" || text == "off" || text == "OFF") {
        return false;
    }
    return default_value;
}

StitchTuning loadStitchTuning() {
    StitchTuning tuning;
    tuning.sift_features = envIntOr("STITCH_SIFT_FEATURES", tuning.sift_features, 128);
    tuning.match_conf = envFloatOr("STITCH_MATCH_CONF", tuning.match_conf, 0.01f);
    tuning.min_good_matches = envIntOr("STITCH_MIN_GOOD_MATCHES", tuning.min_good_matches, 4);
    tuning.min_inliers = envIntOr("STITCH_MIN_INLIERS", tuning.min_inliers, 4);

    tuning.use_range_matcher = envEnabled("STITCH_USE_RANGE_MATCHER", tuning.use_range_matcher);
    tuning.range_width = envIntOr("STITCH_RANGE_WIDTH", tuning.range_width, 2);

    tuning.use_affine_bundle = envEnabled("STITCH_USE_AFFINE_BUNDLE", tuning.use_affine_bundle);
    tuning.use_affine_warper = envEnabled("STITCH_USE_AFFINE_WARPER", tuning.use_affine_warper);
    tuning.use_blocks_gain = envEnabled("STITCH_USE_BLOCKS_GAIN", tuning.use_blocks_gain);
    tuning.blend_bands = envIntOr("STITCH_BLEND_BANDS", tuning.blend_bands, 1);
    tuning.pano_conf_thresh = envFloatOr("STITCH_PANO_CONF", tuning.pano_conf_thresh, 0.01f);
    tuning.use_anchor_fallback = envEnabled("STITCH_USE_ANCHOR_FALLBACK", tuning.use_anchor_fallback);
    tuning.anchor_window = envIntOr("STITCH_ANCHOR_WINDOW", tuning.anchor_window, 1);
    tuning.use_opencl = envEnabled("STITCH_USE_OPENCL", tuning.use_opencl);
    tuning.try_gpu = envEnabled("STITCH_TRY_GPU", tuning.try_gpu);
    tuning.registration_resol_mpx = envFloatOr(
        "STITCH_REGISTRATION_RESOL_MPX", static_cast<float>(tuning.registration_resol_mpx), 0.05f);
    tuning.seam_estimation_resol_mpx = envFloatOr(
        "STITCH_SEAM_RESOL_MPX", static_cast<float>(tuning.seam_estimation_resol_mpx), 0.05f);
    tuning.compositing_resol_mpx = envFloatOr(
        "STITCH_COMPOSITING_RESOL_MPX", static_cast<float>(tuning.compositing_resol_mpx), -1.0f);
    tuning.adaptive_speed = envEnabled("STITCH_ADAPTIVE_SPEED", tuning.adaptive_speed);
    tuning.large_strip_threshold = envIntOr("STITCH_LARGE_STRIP_THRESHOLD", tuning.large_strip_threshold, 2);
    tuning.large_strip_sift_features = envIntOr(
        "STITCH_LARGE_STRIP_SIFT_FEATURES", tuning.large_strip_sift_features, 256);
    tuning.large_strip_range_width = envIntOr("STITCH_LARGE_STRIP_RANGE_WIDTH", tuning.large_strip_range_width, 2);
    return tuning;
}
