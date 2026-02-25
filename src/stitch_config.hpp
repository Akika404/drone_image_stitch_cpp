#pragma once

#include <array>
#include <optional>
#include <string>
#include <vector>

/// 单相机内参/畸变占位。缺失时表示“尚未标定”。
struct CameraCalibration {
    std::string camera_id;   // 例: "visible_ortho"
    int image_width = 0;
    int image_height = 0;

    // 内参（像素单位），未知时保持 nullopt。
    std::optional<double> fx_px;
    std::optional<double> fy_px;
    std::optional<double> cx_px;
    std::optional<double> cy_px;

    // OpenCV 畸变参数顺序：k1,k2,p1,p2,k3,k4,k5,k6。
    std::optional<std::array<double, 8> > distortion;

    // 外参先验（机体->相机），未知时保持 nullopt。
    std::optional<std::array<double, 3> > lever_arm_m;    // dx,dy,dz (m)
    std::optional<std::array<double, 3> > boresight_deg;  // rx,ry,rz (deg)

    [[nodiscard]] bool hasIntrinsics() const {
        return fx_px.has_value() && fy_px.has_value() && cx_px.has_value() && cy_px.has_value();
    }

    [[nodiscard]] bool hasDistortion() const { return distortion.has_value(); }

    [[nodiscard]] bool isMetricReady() const { return hasIntrinsics() && hasDistortion(); }
};

/// 多波段相机组占位。默认全部为空，后续可直接填值启用。
struct MultiBandCalibration {
    std::vector<CameraCalibration> cameras;

    [[nodiscard]] bool anyMetricReady() const {
        for (const auto &cam: cameras) {
            if (cam.isMetricReady()) {
                return true;
            }
        }
        return false;
    }
};

struct StitchTuning {
    /// SIFT 每图最大特征点数，用于 cv::SIFT::create()
    int sift_features = 1500;
    /// strip-stage（行带内部拼接）每图最大特征点数；<=0 时回退到 sift_features
    int strip_sift_features = 1500;
    /// global-stage（行带之间拼接）每图最大特征点数；<=0 时回退到 sift_features
    int global_sift_features = 2500;
    /// Lowe 比率阈值，特征匹配时保留 ratio < match_conf 的匹配，值越大越宽松
    float match_conf = 0.35f;
    /// 配对诊断：最少 good matches 数，低于此值认为无法拼接
    int min_good_matches = 10;
    /// 配对诊断：RANSAC homography 最少内点数，低于此值认为无法拼接
    int min_inliers = 8;

    /// 使用 BestOf2NearestRangeMatcher，按位置限制匹配范围（适合有序航带）
    bool use_range_matcher = true;
    /// 同 strip 内允许匹配的最大位置差；global 模式下 within-strip 配对也用此值
    int range_width = 6;

    /// 使用 BundleAdjusterAffinePartial 做光束法平差
    bool use_affine_bundle = true;
    /// 使用 AffineWarper（否则 PlaneWarper），仿射更适合非平面场景
    bool use_affine_warper = true;
    /// 使用 BlocksGainCompensator 做曝光补偿（否则用 GAIN）
    bool use_blocks_gain = true;
    /// MultiBandBlender 的拉普拉斯金字塔层数，层数越多融合越平滑
    int blend_bands = 5;
    /// leaveBiggestComponent / BundleAdjuster 的置信度阈值，过滤不可靠匹配
    float pano_conf_thresh = 0.7f;

    /// 顺序拼接失败时，是否尝试用 anchor 图像辅助拼接
    bool use_anchor_fallback = false;
    /// 顺序拼接时保留的 anchor 窗口大小（最近 N 张图）
    int anchor_window = 4;

    /// 是否启用 OpenCL
    bool use_opencl = true;
    /// 是否尝试用 GPU 加速 matcher、blender 等
    bool try_gpu = true;

    /// 配准阶段工作分辨率（百万像素），越小特征检测越快
    double registration_resol_mpx = 0.40;
    /// 接缝估计阶段分辨率（百万像素）
    double seam_estimation_resol_mpx = 0.10;
    /// 合成阶段分辨率（百万像素），-1 表示全分辨率
    double compositing_resol_mpx = -1.0;

    /// 标定参数占位：允许先空着，后续拿到参数后直接填。
    MultiBandCalibration calibration;

};

StitchTuning loadStitchTuning(const std::string &image_type = "visible");
