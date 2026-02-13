#include "stitch_robust.hpp"

#include <opencv2/core/ocl.hpp>
#include <opencv2/stitching/detail/blenders.hpp>
#include <opencv2/stitching/detail/exposure_compensate.hpp>
#include <opencv2/stitching/detail/matchers.hpp>
#include <opencv2/stitching/detail/motion_estimators.hpp>
#include <opencv2/stitching/detail/seam_finders.hpp>
#include <opencv2/stitching/warpers.hpp>

#include <algorithm>
#include <deque>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "stitch_common.hpp"

namespace {
    struct PairDiagnostics {
        size_t kp_left = 0;
        size_t kp_right = 0;
        size_t good_matches = 0;
        bool descriptors_ready = false;
        bool homography_ok = false;
        int inliers = 0;
    };

    std::string matInfo(const cv::Mat &img) {
        std::ostringstream oss;
        oss << img.cols << "x" << img.rows << ", ch=" << img.channels();
        return oss.str();
    }

    std::string imageTagAt(const std::vector<std::string> *image_tags, const size_t idx) {
        if (!image_tags || idx >= image_tags->size()) {
            return "img#" + std::to_string(idx);
        }
        return image_tags->at(idx);
    }

    bool looksLikeOpenClFailure(const cv::Exception &e) {
        const std::string msg = e.what();
        return msg.find("OpenCL") != std::string::npos ||
               msg.find("clBuildProgram") != std::string::npos ||
               msg.find("CL_INVALID_COMMAND_QUEUE") != std::string::npos ||
               msg.find("cv::ocl::Program") != std::string::npos;
    }

    void logStitchPhasePlan(const std::string &stage) {
        std::cout << "[" << stage << "] phase begin: feature detection + feature matching" << std::endl;
        std::cout << "[" << stage << "] phase begin: camera parameter estimation" << std::endl;
        std::cout << "[" << stage << "] phase begin: global optimization (bundle adjustment)" << std::endl;
    }

    void logComposePhasePlan(const std::string &stage) {
        std::cout << "[" << stage << "] phase begin: image warping" << std::endl;
        std::cout << "[" << stage << "] phase begin: seam finding" << std::endl;
        std::cout << "[" << stage << "] phase begin: exposure compensation" << std::endl;
        std::cout << "[" << stage << "] phase begin: multi-band blending" << std::endl;
    }

    void logOneShotPairPlan(const std::string &stage_name, const std::vector<std::string> *image_tags) {
        if (!image_tags || image_tags->size() < 2) {
            return;
        }
        for (size_t i = 1; i < image_tags->size(); ++i) {
            std::cout << "[" << stage_name << "] one-shot pair " << i << "/" << (image_tags->size() - 1)
                    << ": " << image_tags->at(i - 1) << " + " << image_tags->at(i) << std::endl;
        }
    }

    PairDiagnostics computePairDiagnostics(const cv::Mat &left, const cv::Mat &right, const int sift_features) {
        PairDiagnostics diag;
        cv::Mat left_gray;
        cv::Mat right_gray;
        if (left.channels() == 1) {
            left_gray = left;
        } else {
            cv::cvtColor(left, left_gray, cv::COLOR_BGR2GRAY);
        }
        if (right.channels() == 1) {
            right_gray = right;
        } else {
            cv::cvtColor(right, right_gray, cv::COLOR_BGR2GRAY);
        }

        cv::Ptr<cv::SIFT> sift = cv::SIFT::create(sift_features);
        std::vector<cv::KeyPoint> kp_left;
        std::vector<cv::KeyPoint> kp_right;
        cv::Mat desc_left;
        cv::Mat desc_right;
        sift->detectAndCompute(left_gray, cv::noArray(), kp_left, desc_left);
        sift->detectAndCompute(right_gray, cv::noArray(), kp_right, desc_right);
        diag.kp_left = kp_left.size();
        diag.kp_right = kp_right.size();

        if (desc_left.empty() || desc_right.empty()) {
            return diag;
        }
        diag.descriptors_ready = true;

        cv::BFMatcher matcher(cv::NORM_L2);
        std::vector<std::vector<cv::DMatch> > knn_matches;
        matcher.knnMatch(desc_left, desc_right, knn_matches, 2);
        std::vector<cv::DMatch> good_matches;
        good_matches.reserve(knn_matches.size());
        for (const auto &m: knn_matches) {
            if (m.size() < 2) {
                continue;
            }
            if (m[0].distance < 0.75f * m[1].distance) {
                good_matches.push_back(m[0]);
            }
        }
        diag.good_matches = good_matches.size();

        if (good_matches.size() < 4) {
            return diag;
        }

        std::vector<cv::Point2f> pts_left;
        std::vector<cv::Point2f> pts_right;
        pts_left.reserve(good_matches.size());
        pts_right.reserve(good_matches.size());
        for (const auto &m: good_matches) {
            pts_left.push_back(kp_left[m.queryIdx].pt);
            pts_right.push_back(kp_right[m.trainIdx].pt);
        }

        cv::Mat inlier_mask;
        cv::Mat H = cv::findHomography(pts_left, pts_right, cv::RANSAC, 3.0, inlier_mask);
        if (H.empty()) {
            return diag;
        }
        diag.homography_ok = true;
        diag.inliers = cv::countNonZero(inlier_mask);
        return diag;
    }

    void logPairDiagnostics(
        const cv::Mat &left,
        const cv::Mat &right,
        const std::string &stage,
        const size_t idx,
        const PairDiagnostics &diag,
        const StitchTuning &tuning) {
        std::cout << "[" << stage << "] failure diagnostics idx=" << idx
                << ", left={" << matInfo(left) << "}, right={" << matInfo(right) << "}"
                << ", kp_left=" << diag.kp_left
                << ", kp_right=" << diag.kp_right;
        if (!diag.descriptors_ready) {
            std::cout << ", desc_empty=true" << std::endl;
            return;
        }
        std::cout << ", good_matches=" << diag.good_matches
                << "(min=" << tuning.min_good_matches << ")";
        if (!diag.homography_ok) {
            if (diag.good_matches < 4) {
                std::cout << ", homography=not_enough_matches" << std::endl;
            } else {
                std::cout << ", homography=failed" << std::endl;
            }
            return;
        }
        std::cout << ", homography=inliers/good_matches="
                << diag.inliers << "/" << diag.good_matches
                << "(min=" << tuning.min_inliers << ")" << std::endl;
    }

    cv::Ptr<cv::Stitcher> createConfiguredStitcher(
        const cv::Stitcher::Mode mode,
        const StitchTuning &tuning,
        const int range_width_override = -1) {
        cv::Ptr<cv::Stitcher> stitcher = cv::Stitcher::create(mode);
        const bool use_gpu_path = tuning.try_gpu;

        stitcher->setPanoConfidenceThresh(tuning.pano_conf_thresh);
        stitcher->setWaveCorrection(false);
        stitcher->setRegistrationResol(tuning.registration_resol_mpx);
        stitcher->setSeamEstimationResol(tuning.seam_estimation_resol_mpx);
        stitcher->setCompositingResol(tuning.compositing_resol_mpx);

        stitcher->setFeaturesFinder(cv::SIFT::create(tuning.sift_features));

        const int range_width = (range_width_override > 0) ? range_width_override : tuning.range_width;
        if (tuning.use_range_matcher && range_width > 1) {
            stitcher->setFeaturesMatcher(
                cv::makePtr<cv::detail::BestOf2NearestRangeMatcher>(range_width, use_gpu_path, tuning.match_conf));
        } else {
            stitcher->setFeaturesMatcher(
                cv::makePtr<cv::detail::BestOf2NearestMatcher>(use_gpu_path, tuning.match_conf));
        }

        if (tuning.use_affine_bundle) {
            stitcher->setBundleAdjuster(cv::makePtr<cv::detail::BundleAdjusterAffinePartial>());
        }

        if (tuning.use_affine_warper) {
            stitcher->setWarper(cv::makePtr<cv::AffineWarper>());
        }

        stitcher->setSeamFinder(cv::makePtr<cv::detail::DpSeamFinder>(cv::detail::DpSeamFinder::COLOR_GRAD));

        if (tuning.use_blocks_gain) {
            stitcher->setExposureCompensator(cv::makePtr<cv::detail::BlocksGainCompensator>());
        }

        stitcher->setBlender(cv::makePtr<cv::detail::MultiBandBlender>(use_gpu_path, tuning.blend_bands));
        return stitcher;
    }

    cv::Stitcher::Status stitchWithMode(
        const std::vector<cv::Mat> &images,
        cv::Mat &output,
        const cv::Stitcher::Mode mode,
        const std::string &stage,
        const StitchTuning &tuning,
        const int range_width_override = -1) {
        if (images.empty()) {
            return cv::Stitcher::ERR_NEED_MORE_IMGS;
        }
        if (images.size() == 1) {
            output = images.front().clone();
            return cv::Stitcher::OK;
        }

        if (images.size() == 2) {
            const auto diag = computePairDiagnostics(images[0], images[1], tuning.sift_features);
            if (!diag.descriptors_ready || static_cast<int>(diag.good_matches) < tuning.min_good_matches) {
                logPairDiagnostics(images[0], images[1], stage, 1, diag, tuning);
                return cv::Stitcher::ERR_HOMOGRAPHY_EST_FAIL;
            }
            if (!diag.homography_ok || diag.inliers < tuning.min_inliers) {
                logPairDiagnostics(images[0], images[1], stage, 1, diag, tuning);
                return cv::Stitcher::ERR_HOMOGRAPHY_EST_FAIL;
            }
        }

        auto run_stitch = [&](const StitchTuning &local_tuning) -> cv::Stitcher::Status {
            cv::Ptr<cv::Stitcher> stitcher = createConfiguredStitcher(mode, local_tuning, range_width_override);
            logStitchPhasePlan(stage);
            const auto estimate_status = stitcher->estimateTransform(images);
            if (estimate_status != cv::Stitcher::OK) {
                return estimate_status;
            }
            logComposePhasePlan(stage);
            return stitcher->composePanorama(output);
        };

        try {
            return run_stitch(tuning);
        } catch (const cv::Exception &e) {
            if (!looksLikeOpenClFailure(e) || !cv::ocl::useOpenCL()) {
                throw;
            }
            std::cerr << "[" << stage << "] OpenCL runtime failure detected, retry on CPU: " << e.what() << std::endl;
            cv::ocl::setUseOpenCL(false);
            StitchTuning cpu_tuning = tuning;
            cpu_tuning.try_gpu = false;
            return run_stitch(cpu_tuning);
        }
    }

    std::optional<cv::Mat> stitchSequentially(
        const std::vector<cv::Mat> &images,
        const cv::Stitcher::Mode mode,
        const std::string &stage_name,
        const StitchTuning &tuning,
        const int range_width_override = -1,
        const std::vector<std::string> *image_tags = nullptr) {
        if (images.empty()) {
            return std::nullopt;
        }
        cv::Mat current = images.front().clone();
        std::deque<cv::Mat> anchors;
        anchors.push_back(images.front());
        const int anchor_window = std::max(1, tuning.anchor_window);

        for (size_t i = 1; i < images.size(); ++i) {
            const std::string left_tag = imageTagAt(image_tags, i - 1);
            const std::string right_tag = imageTagAt(image_tags, i);
            std::cout << "[" << stage_name << "] sequential step " << i << "/" << (images.size() - 1)
                    << ": " << left_tag << " + " << right_tag << std::endl;

            cv::Mat next_result;
            cv::Stitcher::Status status = cv::Stitcher::ERR_HOMOGRAPHY_EST_FAIL;

            if (tuning.use_anchor_fallback && !anchors.empty()) {
                std::vector<cv::Mat> local_batch;
                local_batch.reserve(2 + anchors.size());
                local_batch.push_back(current);
                for (const auto &anchor: anchors) {
                    local_batch.push_back(anchor);
                }
                local_batch.push_back(images[i]);

                const int local_range = std::max(
                    2,
                    std::min(static_cast<int>(local_batch.size()), (range_width_override > 0)
                                                                       ? range_width_override
                                                                       : tuning.range_width));
                status = stitchWithMode(local_batch, next_result, mode, stage_name, tuning, local_range);
            }

            if (status != cv::Stitcher::OK) {
                const std::vector<cv::Mat> pair = {current, images[i]};
                status = stitchWithMode(pair, next_result, mode, stage_name, tuning, range_width_override);
            }

            if (status != cv::Stitcher::OK) {
                std::cout << "[" << stage_name << "] sequential step failed at "
                        << left_tag << " + " << right_tag << std::endl;
                const auto diag = computePairDiagnostics(current, images[i], tuning.sift_features);
                logPairDiagnostics(current, images[i], stage_name, i, diag, tuning);
                return std::nullopt;
            }
            current = next_result;

            anchors.push_back(images[i]);
            while (static_cast<int>(anchors.size()) > anchor_window) {
                anchors.pop_front();
            }
        }
        return current;
    }
} // namespace

cv::Mat stitchRobustly(
    const std::vector<cv::Mat> &images,
    const cv::Stitcher::Mode mode,
    const std::string &stage_name,
    const StitchTuning &tuning,
    const int range_width_override,
    const std::vector<std::string> *image_tags) {
    if (image_tags && image_tags->size() == images.size()) {
        std::cout << "[" << stage_name << "] one-shot stitch begin, images=" << images.size() << std::endl;
        logOneShotPairPlan(stage_name, image_tags);
    } else {
        std::cout << "[" << stage_name << "] one-shot stitch begin, images=" << images.size() << std::endl;
    }

    cv::Mat output;
    const auto first_try_status = stitchWithMode(images, output, mode, stage_name, tuning, range_width_override);
    if (first_try_status == cv::Stitcher::OK) {
        std::cout << "[" << stage_name << "] one-shot stitch success" << std::endl;
        return output;
    }

    std::cout << "[" << stage_name << "] one-shot stitch failed, fallback to sequential: "
            << stitchStatusToString(first_try_status) << std::endl;
    const auto sequential = stitchSequentially(images, mode, stage_name, tuning, range_width_override, image_tags);
    if (sequential.has_value()) {
        return sequential.value();
    }

    throw std::runtime_error(
        "[" + stage_name + "] stitch failed: " + stitchStatusToString(first_try_status) +
        " (code: " + std::to_string(first_try_status) + ")");
}
