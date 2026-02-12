#include "stitch_pipeline.hpp"

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <utility>

using namespace cv;
using namespace cv::detail;
using std::vector;

namespace uav {
    StitchPipeline::StitchPipeline(StitchPipelineConfig config) : config_(std::move(config)) {
    }

    Mat StitchPipeline::stitch(const vector<Mat> &images) const {
        std::cout << "[Stitch] 开始拼接，图像数量: " << images.size() << std::endl;

        if (images.size() < 2) {
            throw std::runtime_error("至少需要两张图像进行拼接");
        }

        if (!config_.featureDetector || !config_.matcher || !config_.estimator ||
            !config_.bundleAdjuster || !config_.warperCreator || !config_.seamFinder ||
            !config_.exposureCompensator || !config_.blender) {
            throw std::runtime_error("拼接配置不完整");
        }

        vector<ImageFeatures> features(images.size());
        for (size_t i = 0; i < images.size(); ++i) {
            Mat gray;
            cvtColor(images[i], gray, COLOR_BGR2GRAY);

            vector<KeyPoint> kps;
            UMat desc;
            config_.featureDetector->detectAndCompute(gray, noArray(), kps, desc);

            features[i].img_idx = static_cast<int>(i);
            features[i].keypoints = kps;
            features[i].descriptors = desc;
            features[i].img_size = images[i].size();

            std::cout << "[Stitch] 图像 " << i << " 特征点数量: " << kps.size() << std::endl;
        }

        std::cout << "[Stitch] 开始进行特征匹配" << std::endl;
        vector<MatchesInfo> pairwise_matches;
        (*config_.matcher)(features, pairwise_matches);
        config_.matcher->collectGarbage();
        std::cout << "[Stitch] 特征匹配完成，匹配对数量: " << pairwise_matches.size() << std::endl;

        std::cout << "[Stitch] 开始估计相机参数" << std::endl;
        vector<CameraParams> cameras;
        if (!(*config_.estimator)(features, pairwise_matches, cameras)) {
            throw std::runtime_error("相机参数估计失败");
        }
        std::cout << "[Stitch] 相机参数估计完成，相机数量: " << cameras.size() << std::endl;

        for (auto &camera: cameras) {
            Mat R;
            camera.R.convertTo(R, CV_32F);
            camera.R = R;
        }

        std::cout << "[Stitch] 开始全局优化(bundle adjustment)" << std::endl;
        config_.bundleAdjuster->setConfThresh(1.0);
        config_.bundleAdjuster->setRefinementMask(Mat::ones(3, 3, CV_8U));
        (*config_.bundleAdjuster)(features, pairwise_matches, cameras);
        std::cout << "[Stitch] 全局优化完成" << std::endl;

        vector<double> focals;
        focals.reserve(cameras.size());
        for (const auto &cam: cameras) {
            focals.push_back(cam.focal);
        }
        std::ranges::sort(focals);
        double warped_focal = focals[focals.size() / 2];
        std::cout << "[Stitch] 选择的投影焦距: " << warped_focal << std::endl;

        Ptr<RotationWarper> warper = config_.warperCreator->create(static_cast<float>(warped_focal));

        vector<Point> corners(images.size());
        vector<UMat> warped(images.size());
        vector<UMat> warped_masks(images.size());
        vector<Size> warped_sizes(images.size());

        std::cout << "[Stitch] 开始图像投影与变换" << std::endl;
        for (size_t i = 0; i < images.size(); ++i) {
            Mat K;
            cameras[i].K().convertTo(K, CV_32F);
            corners[i] = warper->warp(images[i], K, cameras[i].R, INTER_LINEAR, BORDER_REFLECT, warped[i]);

            Mat mask_mat(warped[i].size(), CV_8U, Scalar::all(255));
            mask_mat.copyTo(warped_masks[i]);
            warped_sizes[i] = warped[i].size();

            std::cout << "[Stitch] 图像 " << i << " 变换后尺寸: "
                      << warped_sizes[i].width << "x" << warped_sizes[i].height << std::endl;
        }

        std::cout << "[Stitch] 开始寻找接缝线" << std::endl;
        config_.seamFinder->find(warped, corners, warped_masks);
        std::cout << "[Stitch] 接缝线计算完成" << std::endl;

        std::cout << "[Stitch] 开始曝光补偿" << std::endl;
        config_.exposureCompensator->feed(corners, warped, warped_masks);
        std::cout << "[Stitch] 曝光补偿完成" << std::endl;

        std::cout << "[Stitch] 初始化融合器" << std::endl;
        config_.blender->prepare(corners, warped_sizes);

        std::cout << "[Stitch] 开始融合各幅图像" << std::endl;
        for (size_t i = 0; i < warped.size(); ++i) {
            config_.exposureCompensator->apply(static_cast<int>(i), corners[i], warped[i], warped_masks[i]);
            config_.blender->feed(warped[i], warped_masks[i], corners[i]);
        }

        Mat result, result_mask;
        config_.blender->blend(result, result_mask);
        std::cout << "[Stitch] 融合完成，开始结果归一化" << std::endl;

        Mat result8u;
        if (result.depth() != CV_8U) {
            Mat tmp = result.reshape(1);
            double minv = 0.0, maxv = 0.0;
            minMaxLoc(tmp, &minv, &maxv);
            double scale = maxv > 0.0 ? 255.0 / maxv : 1.0;
            result.convertTo(result8u, CV_8U, scale);
        } else {
            result8u = result;
        }

        std::cout << "[Stitch] 拼接流程结束" << std::endl;

        return result8u;
    }
} // namespace uav
