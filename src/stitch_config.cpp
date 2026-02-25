#include "stitch_config.hpp"

#include <cctype>

namespace {
    std::string normalizeImageType(std::string image_type) {
        std::string normalized;
        normalized.reserve(image_type.size());
        for (const unsigned char c: image_type) {
            if (std::isalnum(c)) {
                normalized.push_back(static_cast<char>(std::tolower(c)));
            }
        }
        return normalized;
    }

    void applyVisiblePreset(StitchTuning &tuning) {
        tuning.sift_features = 2200;
        tuning.strip_sift_features = 2200;
        tuning.global_sift_features = 3600;
        tuning.match_conf = 0.35f;
        tuning.use_range_matcher = true;
        tuning.range_width = 6;
        tuning.use_affine_bundle = true;
        tuning.use_affine_warper = true;
        tuning.blend_bands = 5;
        tuning.registration_resol_mpx = 0.45;
        tuning.seam_estimation_resol_mpx = 0.12;
        tuning.compositing_resol_mpx = -1.0;
    }

    void applyNirPreset(StitchTuning &tuning) {
        tuning.sift_features = 2800;
        tuning.strip_sift_features = 2800;
        tuning.global_sift_features = 4200;
        tuning.match_conf = 0.40f;
        tuning.use_range_matcher = true;
        tuning.range_width = 7;
        tuning.use_affine_bundle = true;
        tuning.use_affine_warper = true;
        tuning.blend_bands = 5;
        tuning.registration_resol_mpx = 0.55;
        tuning.seam_estimation_resol_mpx = 0.15;
        tuning.compositing_resol_mpx = -1.0;
    }

    void applyLwirPreset(StitchTuning &tuning) {
        tuning.sift_features = 900;
        tuning.strip_sift_features = 900;
        tuning.global_sift_features = 1400;
        tuning.match_conf = 0.48f;
        tuning.use_range_matcher = true;
        tuning.range_width = 4;
        tuning.use_affine_bundle = true;
        tuning.use_affine_warper = true;
        tuning.blend_bands = 3;
        tuning.registration_resol_mpx = 0.30;
        tuning.seam_estimation_resol_mpx = 0.08;
        tuning.compositing_resol_mpx = -1.0;
    }

    void initializeCalibrationPlaceholders(StitchTuning &tuning) {
        // 占位模板：参数默认未知（nullopt），后续拿到标定值后直接填充即可。
        CameraCalibration visible;
        visible.camera_id = "visible";
        // 示例：
        // visible.image_width = 2448;
        // visible.image_height = 2048;
        // visible.fx_px = 2500.0; visible.fy_px = 2500.0;
        // visible.cx_px = 1224.0; visible.cy_px = 1024.0;
        // visible.distortion = std::array<double, 8>{k1, k2, p1, p2, k3, k4, k5, k6};
        tuning.calibration.cameras.push_back(visible);

        CameraCalibration nir;
        nir.camera_id = "nir";
        tuning.calibration.cameras.push_back(nir);

        CameraCalibration lwir;
        lwir.camera_id = "lwir";
        tuning.calibration.cameras.push_back(lwir);
    }
}

StitchTuning loadStitchTuning(const std::string &image_type) {
    StitchTuning tuning;
    initializeCalibrationPlaceholders(tuning);

    const std::string normalized = normalizeImageType(image_type);
    if (normalized == "visible" || normalized == "rgb" || normalized == "vis") {
        applyVisiblePreset(tuning);
    } else if (normalized == "nir" || normalized == "nearir" ||
               normalized == "nearinfrared" || normalized == "ninfrared") {
        applyNirPreset(tuning);
    } else if (normalized == "lwir" || normalized == "thermal" || normalized == "long" ||
               normalized == "longwave" || normalized == "longir" || normalized == "tir") {
        applyLwirPreset(tuning);
    } else {
        // 未识别类型默认使用可见光预设，避免空配置。
        applyVisiblePreset(tuning);
    }

    return tuning;
}
