#pragma once

struct StitchTuning {
    /// SIFT 每图最大特征点数，用于 cv::SIFT::create()
    int sift_features = 1500;
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
    bool use_anchor_fallback = true;
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

};

StitchTuning loadStitchTuning();
