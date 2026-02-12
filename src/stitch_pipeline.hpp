#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/stitching/detail/matchers.hpp>
#include <opencv2/stitching/detail/motion_estimators.hpp>
#include <opencv2/stitching/detail/seam_finders.hpp>
#include <opencv2/stitching/detail/exposure_compensate.hpp>
#include <opencv2/stitching/detail/blenders.hpp>

#include <vector>

namespace uav {
    struct StitchPipelineConfig {
        cv::Ptr<cv::Feature2D> featureDetector;
        cv::Ptr<cv::detail::FeaturesMatcher> matcher;
        cv::Ptr<cv::detail::Estimator> estimator;
        cv::Ptr<cv::detail::BundleAdjusterBase> bundleAdjuster;
        cv::Ptr<cv::WarperCreator> warperCreator;
        cv::Ptr<cv::detail::SeamFinder> seamFinder;
        cv::Ptr<cv::detail::ExposureCompensator> exposureCompensator;
        cv::Ptr<cv::detail::Blender> blender;
    };

    class StitchPipeline {
    public:
        explicit StitchPipeline(StitchPipelineConfig config);

        [[nodiscard]] cv::Mat stitch(const std::vector<cv::Mat> &images) const;

    private:
        StitchPipelineConfig config_;
    };
} // namespace uav
