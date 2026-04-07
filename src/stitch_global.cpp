#include "stitch_global.hpp"

#include <opencv2/core/ocl.hpp>
#include <opencv2/stitching/detail/blenders.hpp>
#include <opencv2/stitching/detail/exposure_compensate.hpp>
#include <opencv2/stitching/detail/matchers.hpp>
#include <opencv2/stitching/detail/motion_estimators.hpp>
#include <opencv2/stitching/detail/seam_finders.hpp>
#include <opencv2/stitching/warpers.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>

#include "stitch_common.hpp"

namespace {
    bool looksLikeOpenClFailure(const cv::Exception &e) {
        const std::string msg = e.what();
        return msg.find("OpenCL") != std::string::npos ||
               msg.find("clBuildProgram") != std::string::npos ||
               msg.find("CL_INVALID_COMMAND_QUEUE") != std::string::npos ||
               msg.find("cv::ocl::Program") != std::string::npos ||
               msg.find("AGX") != std::string::npos;
    }

    struct PairAffineEstimate {
        bool ok = false;
        cv::Mat affine_cur_to_ref;
        int good_matches = 0;
        int inliers = 0;
        double inlier_ratio = 0.0;
        double score = -1.0;
    };

    cv::Matx33d affineToMatx33(const cv::Mat &affine_2x3) {
        cv::Matx33d h = cv::Matx33d::eye();
        h(0, 0) = affine_2x3.at<double>(0, 0);
        h(0, 1) = affine_2x3.at<double>(0, 1);
        h(0, 2) = affine_2x3.at<double>(0, 2);
        h(1, 0) = affine_2x3.at<double>(1, 0);
        h(1, 1) = affine_2x3.at<double>(1, 1);
        h(1, 2) = affine_2x3.at<double>(1, 2);
        return h;
    }

    cv::Mat matx33ToAffine(const cv::Matx33d &h) {
        cv::Mat affine = cv::Mat::zeros(2, 3, CV_64F);
        affine.at<double>(0, 0) = h(0, 0);
        affine.at<double>(0, 1) = h(0, 1);
        affine.at<double>(0, 2) = h(0, 2);
        affine.at<double>(1, 0) = h(1, 0);
        affine.at<double>(1, 1) = h(1, 1);
        affine.at<double>(1, 2) = h(1, 2);
        return affine;
    }

    cv::Matx33d translationMatx33(const double tx, const double ty) {
        return {
            1.0, 0.0, tx,
            0.0, 1.0, ty,
            0.0, 0.0, 1.0
        };
    }

    cv::Rect transformedBoundingRect(const cv::Size &size, const cv::Matx33d &h) {
        const std::array<cv::Point2d, 4> corners = {
            cv::Point2d(0.0, 0.0),
            cv::Point2d(static_cast<double>(size.width), 0.0),
            cv::Point2d(static_cast<double>(size.width), static_cast<double>(size.height)),
            cv::Point2d(0.0, static_cast<double>(size.height))
        };

        double min_x = std::numeric_limits<double>::max();
        double min_y = std::numeric_limits<double>::max();
        double max_x = std::numeric_limits<double>::lowest();
        double max_y = std::numeric_limits<double>::lowest();

        for (const auto &p: corners) {
            const cv::Vec3d src(p.x, p.y, 1.0);
            const cv::Vec3d dst = h * src;
            min_x = std::min(min_x, dst[0]);
            min_y = std::min(min_y, dst[1]);
            max_x = std::max(max_x, dst[0]);
            max_y = std::max(max_y, dst[1]);
        }

        const int x = static_cast<int>(std::floor(min_x));
        const int y = static_cast<int>(std::floor(min_y));
        const int w = std::max(1, static_cast<int>(std::ceil(max_x)) - x);
        const int h_px = std::max(1, static_cast<int>(std::ceil(max_y)) - y);
        return {x, y, w, h_px};
    }

    PairAffineEstimate estimatePairAffine(
        const cv::Mat &ref_strip,
        const cv::Mat &cur_strip,
        const StitchTuning &tuning) {
        PairAffineEstimate out;
        if (ref_strip.empty() || cur_strip.empty()) {
            return out;
        }

        cv::Mat ref_gray;
        cv::Mat cur_gray;
        cv::cvtColor(ref_strip, ref_gray, cv::COLOR_BGR2GRAY);
        cv::cvtColor(cur_strip, cur_gray, cv::COLOR_BGR2GRAY);

        cv::Mat ref_mask;
        cv::Mat cur_mask;
        cv::threshold(ref_gray, ref_mask, 2, 255, cv::THRESH_BINARY);
        cv::threshold(cur_gray, cur_mask, 2, 255, cv::THRESH_BINARY);

        const int target_max_side = 2800;
        const double ref_scale = std::min(
            1.0,
            static_cast<double>(target_max_side) /
            static_cast<double>(std::max(ref_strip.cols, ref_strip.rows)));
        const double cur_scale = std::min(
            1.0,
            static_cast<double>(target_max_side) /
            static_cast<double>(std::max(cur_strip.cols, cur_strip.rows)));

        cv::Mat ref_small;
        cv::Mat cur_small;
        cv::Mat ref_mask_small;
        cv::Mat cur_mask_small;
        cv::resize(ref_gray, ref_small, cv::Size(), ref_scale, ref_scale, cv::INTER_AREA);
        cv::resize(cur_gray, cur_small, cv::Size(), cur_scale, cur_scale, cv::INTER_AREA);
        cv::resize(ref_mask, ref_mask_small, ref_small.size(), 0, 0, cv::INTER_NEAREST);
        cv::resize(cur_mask, cur_mask_small, cur_small.size(), 0, 0, cv::INTER_NEAREST);

        const int sift_features = tuning.global_sift_features > 0
                                      ? tuning.global_sift_features
                                      : tuning.sift_features;
        cv::Ptr<cv::Feature2D> sift = cv::SIFT::create(sift_features);

        std::vector<cv::KeyPoint> kp_ref;
        std::vector<cv::KeyPoint> kp_cur;
        cv::Mat desc_ref;
        cv::Mat desc_cur;
        sift->detectAndCompute(ref_small, ref_mask_small, kp_ref, desc_ref);
        sift->detectAndCompute(cur_small, cur_mask_small, kp_cur, desc_cur);
        if (desc_ref.empty() || desc_cur.empty() || kp_ref.size() < 6 || kp_cur.size() < 6) {
            return out;
        }

        cv::BFMatcher matcher(cv::NORM_L2);
        std::vector<std::vector<cv::DMatch> > knn_matches;
        matcher.knnMatch(desc_cur, desc_ref, knn_matches, 2);

        const double ratio_thresh = std::clamp(
            static_cast<double>(tuning.match_conf) + 0.45, 0.65, 0.92);
        std::vector<cv::DMatch> good_matches;
        good_matches.reserve(knn_matches.size());
        for (const auto &m: knn_matches) {
            if (m.size() < 2) {
                continue;
            }
            if (m[0].distance < ratio_thresh * m[1].distance) {
                good_matches.push_back(m[0]);
            }
        }
        out.good_matches = static_cast<int>(good_matches.size());
        const int min_good = std::max(6, tuning.min_good_matches / 2);
        if (out.good_matches < min_good) {
            return out;
        }

        std::vector<cv::Point2f> pts_cur;
        std::vector<cv::Point2f> pts_ref;
        pts_cur.reserve(good_matches.size());
        pts_ref.reserve(good_matches.size());
        for (const auto &m: good_matches) {
            pts_cur.push_back(kp_cur[m.queryIdx].pt);
            pts_ref.push_back(kp_ref[m.trainIdx].pt);
        }

        cv::Mat inlier_mask;
        cv::Mat affine_small = cv::estimateAffine2D(
            pts_cur, pts_ref, inlier_mask, cv::RANSAC, 4.0, 4000, 0.995, 60);
        if (affine_small.empty() || affine_small.rows != 2 || affine_small.cols != 3) {
            return out;
        }

        int inliers = 0;
        for (int i = 0; i < inlier_mask.rows; ++i) {
            if (inlier_mask.at<uchar>(i, 0)) {
                ++inliers;
            }
        }
        out.inliers = inliers;
        out.inlier_ratio = out.good_matches > 0
                               ? static_cast<double>(out.inliers) / static_cast<double>(out.good_matches)
                               : 0.0;
        const int min_inliers = std::max(5, tuning.min_inliers / 2);
        if (out.inliers < min_inliers) {
            return out;
        }

        cv::Mat affine_small_64;
        affine_small.convertTo(affine_small_64, CV_64F);
        cv::Matx33d small_h = affineToMatx33(affine_small_64);
        const cv::Matx33d cur_scale_h(
            cur_scale, 0.0, 0.0,
            0.0, cur_scale, 0.0,
            0.0, 0.0, 1.0);
        const cv::Matx33d ref_inv_scale_h(
            1.0 / ref_scale, 0.0, 0.0,
            0.0, 1.0 / ref_scale, 0.0,
            0.0, 0.0, 1.0);
        const cv::Matx33d full_h = ref_inv_scale_h * small_h * cur_scale_h;
        out.affine_cur_to_ref = matx33ToAffine(full_h);
        out.score = static_cast<double>(out.inliers) * 1.0 +
                    out.inlier_ratio * 20.0 +
                    static_cast<double>(out.good_matches) * 0.02;
        out.ok = true;
        return out;
    }

    std::vector<cv::Rect> buildStripRoiCandidates(const cv::Size &sz) {
        auto make_roi = [&](const double x0, const double x1, const double y0, const double y1) {
            const int x = std::clamp(static_cast<int>(std::floor(sz.width * x0)), 0, std::max(0, sz.width - 1));
            const int y = std::clamp(static_cast<int>(std::floor(sz.height * y0)), 0, std::max(0, sz.height - 1));
            const int r = std::clamp(static_cast<int>(std::ceil(sz.width * x1)), x + 1, sz.width);
            const int b = std::clamp(static_cast<int>(std::ceil(sz.height * y1)), y + 1, sz.height);
            return cv::Rect(x, y, r - x, b - y);
        };

        std::vector<cv::Rect> rois;
        rois.push_back(make_roi(0.00, 1.00, 0.00, 1.00)); // full
        rois.push_back(make_roi(0.00, 0.68, 0.05, 0.95)); // left-heavy
        rois.push_back(make_roi(0.32, 1.00, 0.05, 0.95)); // right-heavy
        rois.push_back(make_roi(0.16, 0.84, 0.05, 0.95)); // center

        // De-duplicate by rect tuple.
        std::vector<cv::Rect> dedup;
        for (const auto &r: rois) {
            if (r.width < 120 || r.height < 120) {
                continue;
            }
            bool exists = false;
            for (const auto &d: dedup) {
                if (d.x == r.x && d.y == r.y && d.width == r.width && d.height == r.height) {
                    exists = true;
                    break;
                }
            }
            if (!exists) {
                dedup.push_back(r);
            }
        }
        return dedup;
    }

    PairAffineEstimate estimatePairAffineWithRoiSearch(
        const cv::Mat &ref_strip,
        const cv::Mat &cur_strip,
        const StitchTuning &tuning) {
        PairAffineEstimate best;
        const std::vector<cv::Rect> ref_rois = buildStripRoiCandidates(ref_strip.size());
        const std::vector<cv::Rect> cur_rois = buildStripRoiCandidates(cur_strip.size());

        for (const auto &ref_roi: ref_rois) {
            for (const auto &cur_roi: cur_rois) {
                const cv::Mat ref_patch = ref_strip(ref_roi);
                const cv::Mat cur_patch = cur_strip(cur_roi);
                PairAffineEstimate local = estimatePairAffine(ref_patch, cur_patch, tuning);
                if (!local.ok) {
                    continue;
                }
                const cv::Matx33d full_h =
                    translationMatx33(static_cast<double>(ref_roi.x), static_cast<double>(ref_roi.y)) *
                    affineToMatx33(local.affine_cur_to_ref) *
                    translationMatx33(static_cast<double>(-cur_roi.x), static_cast<double>(-cur_roi.y));
                local.affine_cur_to_ref = matx33ToAffine(full_h);
                if (!best.ok || local.score > best.score) {
                    best = std::move(local);
                }
            }
        }

        return best;
    }

    void applyChannelGainInPlace(cv::UMat &image, const cv::Vec3d &gain) {
        cv::Mat image_host;
        image.copyTo(image_host);

        cv::Mat image_f32;
        image_host.convertTo(image_f32, CV_32F);
        std::vector<cv::Mat> channels;
        cv::split(image_f32, channels);
        for (int c = 0; c < 3; ++c) {
            channels[c] *= static_cast<float>(gain[c]);
        }
        cv::merge(channels, image_f32);
        image_f32.convertTo(image_host, CV_8U);
        image_host.copyTo(image);
    }

    cv::Ptr<cv::detail::ExposureCompensator> makeSafeExposureCompensator(
        const double canvas_area_mpx,
        std::string &mode_out) {
        if (canvas_area_mpx < 0.0) {
            mode_out = "NO";
            return cv::detail::ExposureCompensator::createDefault(cv::detail::ExposureCompensator::NO);
        }

        if (canvas_area_mpx <= 120.0) {
            auto channels = cv::makePtr<cv::detail::ChannelsCompensator>(2);
            channels->setSimilarityThreshold(0.95);
            mode_out = "CHANNELS";
            return channels;
        }

        auto gain = cv::makePtr<cv::detail::GainCompensator>(1);
        gain->setSimilarityThreshold(0.95);
        mode_out = "GAIN";
        return gain;
    }

    void ensureBinaryMask(cv::UMat &mask) {
        cv::threshold(mask, mask, 1.0, 255.0, cv::THRESH_BINARY);
    }

    cv::UMat buildSoftBlendMask(const cv::UMat &seam_mask, const cv::UMat &content_mask) {
        cv::UMat binary_mask;
        cv::bitwise_and(seam_mask, content_mask, binary_mask);
        ensureBinaryMask(binary_mask);

        cv::Mat binary_host;
        binary_mask.copyTo(binary_host);

        cv::Mat binary_f32;
        binary_host.convertTo(binary_f32, CV_32F, 1.0 / 255.0);

        cv::Mat soft_f32;
        // Feather the seam transition a bit so strip boundaries do not remain as hard lines.
        cv::GaussianBlur(binary_f32, soft_f32, cv::Size(0, 0), 10.0, 10.0, cv::BORDER_REPLICATE);
        cv::multiply(soft_f32, binary_f32, soft_f32);

        cv::Mat soft_u8;
        soft_f32.convertTo(soft_u8, CV_8U, 255.0);
        return soft_u8.getUMat(cv::ACCESS_READ);
    }

    cv::Mat buildWarpedContentMask(
        const cv::Mat &src_image,
        const cv::Mat &affine,
        const cv::Size &dst_size) {
        cv::Mat gray;
        cv::cvtColor(src_image, gray, cv::COLOR_BGR2GRAY);

        cv::Mat content_mask_u8;
        // Strip panoramas may still contain black background wedges after rectangular cropping.
        // Treat only non-black source pixels as valid so those holes are excluded in global blending.
        cv::threshold(gray, content_mask_u8, 3, 255, cv::THRESH_BINARY);

        cv::Mat content_mask_f32;
        content_mask_u8.convertTo(content_mask_f32, CV_32F, 1.0 / 255.0);

        cv::Mat warped_content_f32;
        cv::warpAffine(
            content_mask_f32,
            warped_content_f32,
            affine,
            dst_size,
            cv::INTER_LINEAR,
            cv::BORDER_CONSTANT,
            cv::Scalar(0.0f));

        cv::Mat mask_warped;
        // Only keep pixels whose bilinear footprint comes entirely from valid source content.
        cv::threshold(warped_content_f32, mask_warped, 0.999f, 255.0, cv::THRESH_BINARY);
        mask_warped.convertTo(mask_warped, CV_8U);
        return mask_warped;
    }
}

cv::Mat stitchInterStripsCustom(
    const std::vector<cv::Mat> &strip_panoramas,
    const StitchTuning &tuning) {
    const std::string stage = "GlobalCustom";
    const auto stage_start = std::chrono::steady_clock::now();
    const int num_strips = static_cast<int>(strip_panoramas.size());
    if (num_strips < 2) {
        throw std::runtime_error("[" + stage + "] need >= 2 strip panoramas");
    }

    std::vector<cv::Mat> oriented(num_strips);
    oriented[0] = strip_panoramas[0];
    std::vector<cv::Matx33d> global_transforms(num_strips, cv::Matx33d::eye());

    std::cout << "[" << stage << "] pairwise strip alignment begin, strips=" << num_strips << std::endl;
    for (int i = 1; i < num_strips; ++i) {
        const cv::Mat &ref = oriented[i - 1];
        const cv::Mat &cur = strip_panoramas[i];

        const PairAffineEstimate direct = estimatePairAffineWithRoiSearch(ref, cur, tuning);
        cv::Mat cur_flipped;
        cv::flip(cur, cur_flipped, 1);
        const PairAffineEstimate flipped = estimatePairAffineWithRoiSearch(ref, cur_flipped, tuning);

        const bool choose_flipped = (!direct.ok && flipped.ok) ||
                                    (direct.ok && flipped.ok && (
                                         flipped.inliers > direct.inliers ||
                                         (flipped.inliers == direct.inliers &&
                                          flipped.inlier_ratio > direct.inlier_ratio)));

        PairAffineEstimate best = direct;
        oriented[i] = cur;
        if (choose_flipped) {
            best = flipped;
            oriented[i] = cur_flipped;
        }
        if (!best.ok) {
            throw std::runtime_error(
                "[" + stage + "] strip pair " + std::to_string(i - 1) + "->" + std::to_string(i) +
                " alignment failed (direct matches/inliers=" + std::to_string(direct.good_matches) + "/" +
                std::to_string(direct.inliers) + ", flipped=" + std::to_string(flipped.good_matches) + "/" +
                std::to_string(flipped.inliers) + ")");
        }

        global_transforms[i] = global_transforms[i - 1] * affineToMatx33(best.affine_cur_to_ref);
        std::cout << "[" << stage << "] strip " << i - 1 << "->" << i
                << " aligned: matches=" << best.good_matches
                << ", inliers=" << best.inliers
                << ", ratio=" << best.inlier_ratio
                << ", score=" << best.score
                << ", flipped=" << (choose_flipped ? "yes" : "no") << std::endl;
    }

    int min_x = std::numeric_limits<int>::max();
    int min_y = std::numeric_limits<int>::max();
    int max_x = std::numeric_limits<int>::lowest();
    int max_y = std::numeric_limits<int>::lowest();
    std::vector<cv::Rect> pre_shift_rois(num_strips);
    for (int i = 0; i < num_strips; ++i) {
        pre_shift_rois[i] = transformedBoundingRect(oriented[i].size(), global_transforms[i]);
        min_x = std::min(min_x, pre_shift_rois[i].x);
        min_y = std::min(min_y, pre_shift_rois[i].y);
        max_x = std::max(max_x, pre_shift_rois[i].x + pre_shift_rois[i].width);
        max_y = std::max(max_y, pre_shift_rois[i].y + pre_shift_rois[i].height);
    }

    const int canvas_w = std::max(1, max_x - min_x);
    const int canvas_h = std::max(1, max_y - min_y);
    const cv::Matx33d shift(
        1.0, 0.0, static_cast<double>(-min_x),
        0.0, 1.0, static_cast<double>(-min_y),
        0.0, 0.0, 1.0);
    std::cout << "[" << stage << "] canvas: " << canvas_w << "x" << canvas_h << std::endl;

    std::vector<cv::Point> corners(num_strips);
    std::vector<cv::Size> sizes(num_strips);
    std::vector<cv::Matx33d> shifted_transforms(num_strips);
    for (int i = 0; i < num_strips; ++i) {
        shifted_transforms[i] = shift * global_transforms[i];
        const cv::Rect roi = transformedBoundingRect(oriented[i].size(), shifted_transforms[i]);
        corners[i] = roi.tl();
        sizes[i] = roi.size();
    }

    std::cout << "[" << stage << "] warping strips..." << std::endl;
    std::vector<cv::UMat> warped_imgs(num_strips);
    std::vector<cv::UMat> warped_masks(num_strips);
    for (int i = 0; i < num_strips; ++i) {
        cv::Mat affine = matx33ToAffine(shifted_transforms[i]);
        affine.at<double>(0, 2) -= static_cast<double>(corners[i].x);
        affine.at<double>(1, 2) -= static_cast<double>(corners[i].y);

        cv::Mat warped;
        cv::warpAffine(
            oriented[i], warped, affine, sizes[i], cv::INTER_LINEAR, cv::BORDER_CONSTANT);
        cv::Mat mask_warped = buildWarpedContentMask(oriented[i], affine, sizes[i]);

        warped.copyTo(warped_imgs[i]);
        mask_warped.copyTo(warped_masks[i]);
        std::cout << "[" << stage << "]   warped " << (i + 1) << "/" << num_strips << std::endl;
    }

    const double canvas_area_mpx = static_cast<double>(canvas_w) * static_cast<double>(canvas_h) / 1e6;
    const bool ocl_was_on = cv::ocl::useOpenCL();
    if (ocl_was_on) {
        std::cout << "[" << stage << "] disabling OpenCL for compose phase (canvas="
                << canvas_area_mpx << " MP, strips=" << num_strips
                << ") to avoid AGX compiled-variants limit" << std::endl;
        cv::ocl::setUseOpenCL(false);
    }

    std::cout << "[" << stage << "] pre-equalizing strip radiometry..." << std::endl;
    {
        std::vector<cv::Vec3d> cum_gain(num_strips, cv::Vec3d(1.0, 1.0, 1.0));
        for (int i = 1; i < num_strips; ++i) {
            const cv::Rect rect_prev(corners[i - 1],
                cv::Size(warped_imgs[i - 1].cols, warped_imgs[i - 1].rows));
            const cv::Rect rect_curr(corners[i],
                cv::Size(warped_imgs[i].cols, warped_imgs[i].rows));
            cv::Rect overlap = rect_prev & rect_curr;

            if (overlap.area() < 100) {
                cum_gain[i] = cum_gain[i - 1];
                std::cout << "[" << stage << "]   strips " << (i - 1) << "->" << i
                        << " no overlap, inheriting previous gain" << std::endl;
                continue;
            }

            cv::Rect local_prev(overlap.tl() - corners[i - 1], overlap.size());
            cv::Rect local_curr(overlap.tl() - corners[i], overlap.size());
            local_prev &= cv::Rect(0, 0, warped_imgs[i - 1].cols, warped_imgs[i - 1].rows);
            local_curr &= cv::Rect(0, 0, warped_imgs[i].cols, warped_imgs[i].rows);
            const int w = std::min(local_prev.width, local_curr.width);
            const int h = std::min(local_prev.height, local_curr.height);
            if (w < 10 || h < 10) { cum_gain[i] = cum_gain[i - 1]; continue; }
            local_prev.width = w; local_prev.height = h;
            local_curr.width = w; local_curr.height = h;

            cv::Mat mask_prev_m, mask_curr_m, combined_mask;
            warped_masks[i - 1](local_prev).copyTo(mask_prev_m);
            warped_masks[i](local_curr).copyTo(mask_curr_m);
            cv::bitwise_and(mask_prev_m, mask_curr_m, combined_mask);
            const int valid_px = cv::countNonZero(combined_mask);
            if (valid_px < 1000) { cum_gain[i] = cum_gain[i - 1]; continue; }

            cv::Mat img_prev_m, img_curr_m;
            warped_imgs[i - 1](local_prev).copyTo(img_prev_m);
            warped_imgs[i](local_curr).copyTo(img_curr_m);
            const cv::Scalar mean_prev = cv::mean(img_prev_m, combined_mask);
            const cv::Scalar mean_curr = cv::mean(img_curr_m, combined_mask);

            cv::Vec3d pw_gain(1.0, 1.0, 1.0);
            for (int c = 0; c < 3; ++c) {
                if (mean_curr[c] > 5.0 && mean_prev[c] > 5.0)
                    pw_gain[c] = std::clamp(mean_prev[c] / mean_curr[c], 0.80, 1.25);
            }
            for (int c = 0; c < 3; ++c)
                cum_gain[i][c] = cum_gain[i - 1][c] * pw_gain[c];

            std::cout << "[" << stage << "]   strip " << i
                    << " pw_gain=[" << pw_gain[0] << "," << pw_gain[1] << "," << pw_gain[2] << "]"
                    << " cum=[" << cum_gain[i][0] << "," << cum_gain[i][1] << "," << cum_gain[i][2] << "]"
                    << " overlap=" << valid_px << "px" << std::endl;
        }

        cv::Vec3d geo_mean(1.0, 1.0, 1.0);
        for (int i = 0; i < num_strips; ++i)
            for (int c = 0; c < 3; ++c)
                geo_mean[c] *= cum_gain[i][c];
        for (int c = 0; c < 3; ++c)
            geo_mean[c] = std::pow(geo_mean[c], 1.0 / num_strips);

        for (int i = 0; i < num_strips; ++i) {
            for (int c = 0; c < 3; ++c)
                if (geo_mean[c] > 0.01) cum_gain[i][c] /= geo_mean[c];

            const cv::Vec3d &g = cum_gain[i];
            if (std::abs(g[0] - 1.0) < 0.02 && std::abs(g[1] - 1.0) < 0.02 &&
                std::abs(g[2] - 1.0) < 0.02)
                continue;

            applyChannelGainInPlace(warped_imgs[i], g);

            std::cout << "[" << stage << "]   applied radiometric gain to strip " << i
                    << ": [" << g[0] << "," << g[1] << "," << g[2] << "]" << std::endl;
        }
    }
    std::cout << "[" << stage << "] pre-equalization done" << std::endl;

    std::string exposure_mode;
    cv::Ptr<cv::detail::ExposureCompensator> compensator =
        makeSafeExposureCompensator(canvas_area_mpx, exposure_mode);
    std::cout << "[" << stage << "] exposure compensation begin, mode=" << exposure_mode
            << ", canvas_mpx=" << canvas_area_mpx << std::endl;
    compensator->feed(corners, warped_imgs, warped_masks);
    std::cout << "[" << stage << "] exposure compensation done" << std::endl;

    std::cout << "[" << stage << "] seam finding begin..." << std::endl;
    std::vector<cv::UMat> seam_masks(num_strips);
    const double seam_target_mpx = 8.0;
    const double seam_scale = std::min(
        1.0,
        std::sqrt((seam_target_mpx * 1e6) /
                  (static_cast<double>(canvas_w) * static_cast<double>(canvas_h))));
    std::cout << "[" << stage << "] seam scale=" << seam_scale << std::endl;

    std::vector<cv::UMat> seam_images_f32(num_strips);
    std::vector<cv::Point> seam_corners = corners;
    if (seam_scale < 0.999) {
        for (auto &p: seam_corners) {
            p.x = cvRound(p.x * seam_scale);
            p.y = cvRound(p.y * seam_scale);
        }
        for (int i = 0; i < num_strips; ++i) {
            cv::Mat small_img;
            cv::resize(warped_imgs[i], small_img, cv::Size(), seam_scale, seam_scale, cv::INTER_LINEAR_EXACT);
            cv::Mat small_mask;
            cv::resize(warped_masks[i], small_mask, small_img.size(), 0, 0, cv::INTER_NEAREST);
            small_img.convertTo(seam_images_f32[i], CV_32F);
            small_mask.copyTo(seam_masks[i]);
            ensureBinaryMask(seam_masks[i]);
        }
    } else {
        for (int i = 0; i < num_strips; ++i) {
            seam_masks[i] = warped_masks[i].clone();
            ensureBinaryMask(seam_masks[i]);
            warped_imgs[i].convertTo(seam_images_f32[i], CV_32F);
        }
    }

    try {
        auto seam_finder = cv::makePtr<cv::detail::GraphCutSeamFinder>(
            cv::detail::GraphCutSeamFinderBase::COST_COLOR_GRAD);
        seam_finder->find(seam_images_f32, seam_corners, seam_masks);
        std::cout << "[" << stage << "] seam finder: GraphCut(COLOR_GRAD)" << std::endl;
    } catch (const cv::Exception &e) {
        std::cerr << "[" << stage << "] seam finder GraphCut failed, fallback to DpSeamFinder: "
                << e.what() << std::endl;
        auto seam_finder = cv::makePtr<cv::detail::DpSeamFinder>(cv::detail::DpSeamFinder::COLOR_GRAD);
        seam_finder->find(seam_images_f32, seam_corners, seam_masks);
        std::cout << "[" << stage << "] seam finder: DpSeamFinder(COLOR_GRAD)" << std::endl;
    }
    seam_images_f32.clear();

    std::cout << "[" << stage << "] seam finding done" << std::endl;

    const int auto_blend_bands = std::min(12,
        static_cast<int>(std::ceil(std::log2(
            static_cast<double>(std::max(canvas_w, canvas_h))))) - 1);
    const int final_blend_bands = std::max(std::max(5, tuning.blend_bands), auto_blend_bands);
    cv::Ptr<cv::detail::Blender> blender = cv::makePtr<cv::detail::MultiBandBlender>(
        false, final_blend_bands);
    blender->prepare(corners, sizes);
    std::cout << "[" << stage << "] blender prepared, blend_bands="
            << final_blend_bands << " (config=" << tuning.blend_bands
            << ", auto=" << auto_blend_bands << ")" << std::endl;

    for (int i = 0; i < num_strips; ++i) {
        compensator->apply(i, corners[i], warped_imgs[i], warped_masks[i]);

        cv::UMat warped_16s;
        warped_imgs[i].convertTo(warped_16s, CV_16S);

        cv::UMat blend_mask;
        if (seam_scale < 0.999) {
            cv::resize(seam_masks[i], blend_mask, warped_masks[i].size(), 0, 0, cv::INTER_NEAREST);
        } else {
            blend_mask = seam_masks[i];
        }
        ensureBinaryMask(blend_mask);
        blend_mask = buildSoftBlendMask(blend_mask, warped_masks[i]);

        blender->feed(warped_16s, blend_mask, corners[i]);
        std::cout << "[" << stage << "]   blender feed " << (i + 1) << "/" << num_strips << std::endl;
    }

    std::cout << "[" << stage << "] blending..." << std::endl;
    cv::Mat result;
    cv::Mat result_mask;
    blender->blend(result, result_mask);
    result.convertTo(result, CV_8U);
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - stage_start);
    std::cout << "[" << stage << "] panorama: " << result.cols << "x" << result.rows << std::endl;
    std::cout << "[" << stage << "] done in " << (elapsed.count() / 1000.0) << "s" << std::endl;
    if (ocl_was_on) {
        cv::ocl::setUseOpenCL(true);
    }
    return result;
}
