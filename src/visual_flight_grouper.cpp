#include "visual_flight_grouper.hpp"

#include <opencv2/features2d.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {
    struct ConsecutiveRelation {
        bool ok = false;
        size_t kp_left = 0;
        size_t kp_right = 0;
        int good_matches = 0;
        int inliers = 0;
        double inlier_ratio = 0.0;
        double tx = 0.0;
        double ty = 0.0;
        double scale = 1.0;
        double rotation_deg = 0.0;
        double score = -1.0;
    };

    struct MotionStats {
        bool valid = false;
        bool dominant_horizontal = true;
        double median_main = 0.0;
        double median_cross = 0.0;
        double stable_min_main = 0.0;
        double stable_max_cross = 0.0;
        double duplicate_max_main = 0.0;
        double duplicate_max_cross = 0.0;
    };

    double medianOf(std::vector<double> values) {
        if (values.empty()) {
            return 0.0;
        }
        const size_t mid = values.size() / 2;
        std::nth_element(values.begin(), values.begin() + static_cast<long>(mid), values.end());
        double median = values[mid];
        if (values.size() % 2 == 0 && mid > 0) {
            const auto max_lower = *std::max_element(values.begin(), values.begin() + static_cast<long>(mid));
            median = 0.5 * (median + max_lower);
        }
        return median;
    }

    cv::Mat ensureGray(const cv::Mat &image) {
        if (image.channels() == 1) {
            return image;
        }
        cv::Mat gray;
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        return gray;
    }

    cv::Matx33d scaleMat(const double s) {
        return {
            s, 0.0, 0.0,
            0.0, s, 0.0,
            0.0, 0.0, 1.0
        };
    }

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

    ConsecutiveRelation estimateRelation(
        const cv::Mat &left,
        const cv::Mat &right,
        const StitchTuning &tuning) {
        ConsecutiveRelation relation;
        if (left.empty() || right.empty()) {
            return relation;
        }

        const cv::Mat left_gray = ensureGray(left);
        const cv::Mat right_gray = ensureGray(right);

        constexpr int target_max_side = 1800;
        const double left_scale = std::min(
            1.0,
            static_cast<double>(target_max_side) /
            static_cast<double>(std::max(left.cols, left.rows)));
        const double right_scale = std::min(
            1.0,
            static_cast<double>(target_max_side) /
            static_cast<double>(std::max(right.cols, right.rows)));

        cv::Mat left_small;
        cv::Mat right_small;
        cv::resize(left_gray, left_small, cv::Size(), left_scale, left_scale, cv::INTER_AREA);
        cv::resize(right_gray, right_small, cv::Size(), right_scale, right_scale, cv::INTER_AREA);

        const int sift_features = std::max(
            600,
            std::min(1800, tuning.strip_sift_features > 0 ? tuning.strip_sift_features : tuning.sift_features));
        cv::Ptr<cv::SIFT> sift = cv::SIFT::create(sift_features);

        std::vector<cv::KeyPoint> kp_left;
        std::vector<cv::KeyPoint> kp_right;
        cv::Mat desc_left;
        cv::Mat desc_right;
        sift->detectAndCompute(left_small, cv::noArray(), kp_left, desc_left);
        sift->detectAndCompute(right_small, cv::noArray(), kp_right, desc_right);
        relation.kp_left = kp_left.size();
        relation.kp_right = kp_right.size();

        if (desc_left.empty() || desc_right.empty()) {
            return relation;
        }

        cv::BFMatcher matcher(cv::NORM_L2);
        std::vector<std::vector<cv::DMatch> > knn_matches;
        matcher.knnMatch(desc_right, desc_left, knn_matches, 2);

        const double ratio_thresh = std::clamp(
            static_cast<double>(tuning.match_conf) + 0.45,
            0.65,
            0.92);
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
        relation.good_matches = static_cast<int>(good_matches.size());
        if (relation.good_matches < std::max(8, tuning.min_good_matches / 2)) {
            return relation;
        }

        std::vector<cv::Point2f> pts_right;
        std::vector<cv::Point2f> pts_left;
        pts_right.reserve(good_matches.size());
        pts_left.reserve(good_matches.size());
        for (const auto &match: good_matches) {
            pts_right.push_back(kp_right[match.queryIdx].pt);
            pts_left.push_back(kp_left[match.trainIdx].pt);
        }

        cv::Mat inlier_mask;
        cv::Mat affine_small = cv::estimateAffinePartial2D(
            pts_right, pts_left, inlier_mask, cv::RANSAC, 4.0, 4000, 0.995, 80);
        if (affine_small.empty()) {
            return relation;
        }

        relation.inliers = cv::countNonZero(inlier_mask);
        if (relation.inliers < std::max(6, tuning.min_inliers / 2)) {
            return relation;
        }
        relation.inlier_ratio = static_cast<double>(relation.inliers) /
                                static_cast<double>(std::max(1, relation.good_matches));

        cv::Mat affine_small_64;
        affine_small.convertTo(affine_small_64, CV_64F);
        const cv::Matx33d full_affine =
            cv::Matx33d(scaleMat(1.0 / left_scale)) *
            affineToMatx33(affine_small_64) *
            cv::Matx33d(scaleMat(right_scale));

        const double a00 = full_affine(0, 0);
        const double a01 = full_affine(0, 1);
        const double a10 = full_affine(1, 0);
        const double a11 = full_affine(1, 1);
        const double scale_x = std::sqrt(a00 * a00 + a10 * a10);
        const double scale_y = std::sqrt(a01 * a01 + a11 * a11);
        relation.scale = 0.5 * (scale_x + scale_y);
        relation.rotation_deg = std::atan2(a10, a00) * 180.0 / CV_PI;
        relation.tx = full_affine(0, 2);
        relation.ty = full_affine(1, 2);
        relation.score = static_cast<double>(relation.inliers) +
                         relation.inlier_ratio * 20.0 +
                         static_cast<double>(relation.good_matches) * 0.02;

        if (relation.inlier_ratio < 0.28) {
            return relation;
        }
        if (relation.scale < 0.80 || relation.scale > 1.20) {
            return relation;
        }
        if (std::abs(relation.rotation_deg) > 25.0) {
            return relation;
        }

        relation.ok = true;
        return relation;
    }

    MotionStats summarizeMotion(const std::vector<ConsecutiveRelation> &relations) {
        MotionStats stats;
        std::vector<double> abs_tx;
        std::vector<double> abs_ty;
        for (const auto &relation: relations) {
            if (!relation.ok) {
                continue;
            }
            abs_tx.push_back(std::abs(relation.tx));
            abs_ty.push_back(std::abs(relation.ty));
        }

        if (abs_tx.size() < 2) {
            return stats;
        }

        stats.valid = true;
        stats.dominant_horizontal = medianOf(abs_tx) >= medianOf(abs_ty);

        std::vector<double> abs_main;
        std::vector<double> abs_cross;
        abs_main.reserve(abs_tx.size());
        abs_cross.reserve(abs_tx.size());
        for (const auto &relation: relations) {
            if (!relation.ok) {
                continue;
            }
            if (stats.dominant_horizontal) {
                abs_main.push_back(std::abs(relation.tx));
                abs_cross.push_back(std::abs(relation.ty));
            } else {
                abs_main.push_back(std::abs(relation.ty));
                abs_cross.push_back(std::abs(relation.tx));
            }
        }

        stats.median_main = medianOf(abs_main);
        stats.median_cross = medianOf(abs_cross);
        stats.stable_min_main = std::max(20.0, stats.median_main * 0.30);
        stats.stable_max_cross = std::max(40.0, stats.median_cross * 3.20 + 10.0);
        stats.duplicate_max_main = std::max(10.0, stats.median_main * 0.15);
        stats.duplicate_max_cross = std::max(10.0, stats.median_cross * 1.80 + 5.0);
        return stats;
    }

    double mainMotion(const ConsecutiveRelation &relation, const MotionStats &stats) {
        return stats.dominant_horizontal ? relation.tx : relation.ty;
    }

    double crossMotion(const ConsecutiveRelation &relation, const MotionStats &stats) {
        return stats.dominant_horizontal ? relation.ty : relation.tx;
    }

    bool isDuplicateFrame(const ConsecutiveRelation &relation, const MotionStats &stats) {
        if (!relation.ok) {
            return false;
        }
        return std::abs(mainMotion(relation, stats)) <= stats.duplicate_max_main &&
               std::abs(crossMotion(relation, stats)) <= stats.duplicate_max_cross;
    }

    bool isStableStripStep(const ConsecutiveRelation &relation, const MotionStats &stats) {
        if (!relation.ok) {
            return false;
        }
        return std::abs(mainMotion(relation, stats)) >= stats.stable_min_main &&
               std::abs(crossMotion(relation, stats)) <= stats.stable_max_cross &&
               relation.scale >= 0.85 &&
               relation.scale <= 1.15 &&
               std::abs(relation.rotation_deg) <= 18.0;
    }

    double boundaryMergeScore(const ConsecutiveRelation &relation, const MotionStats &stats) {
        if (!relation.ok) {
            return -1e9;
        }

        double score = relation.score;
        if (isStableStripStep(relation, stats)) {
            score += 1000.0;
        }

        return score;
    }

    void appendGroup(VisualStripGroup &dst, const VisualStripGroup &src) {
        dst.images.insert(dst.images.end(), src.images.begin(), src.images.end());
        dst.image_ids.insert(dst.image_ids.end(), src.image_ids.begin(), src.image_ids.end());
    }

    void prependGroup(VisualStripGroup &dst, const VisualStripGroup &src) {
        dst.images.insert(dst.images.begin(), src.images.begin(), src.images.end());
        dst.image_ids.insert(dst.image_ids.begin(), src.image_ids.begin(), src.image_ids.end());
    }
}

std::vector<VisualStripGroup> VisualFlightGrouper::groupBoustrophedon(
    const std::vector<cv::Mat> &images,
    const std::vector<std::string> &image_ids,
    const StitchTuning &tuning) {
    if (images.empty()) {
        return {};
    }
    if (images.size() != image_ids.size()) {
        throw std::runtime_error("image_ids must align with images");
    }
    if (images.size() == 1) {
        return {{images, image_ids}};
    }

    std::cout << "[VisualGroup] analyzing " << images.size() << " sequential image pairs..." << std::endl;

    std::vector<ConsecutiveRelation> relations;
    relations.reserve(images.size() - 1);
    for (size_t i = 0; i + 1 < images.size(); ++i) {
        const auto relation = estimateRelation(images[i], images[i + 1], tuning);
        relations.push_back(relation);
        std::cout << "[VisualGroup] pair " << i << "->" << (i + 1)
                << " (" << image_ids[i] << " -> " << image_ids[i + 1] << ")"
                << ": ok=" << (relation.ok ? "yes" : "no")
                << ", kp=" << relation.kp_left << "/" << relation.kp_right
                << ", matches=" << relation.good_matches
                << ", inliers=" << relation.inliers
                << ", tx=" << relation.tx
                << ", ty=" << relation.ty
                << ", scale=" << relation.scale
                << ", rot=" << relation.rotation_deg
                << std::endl;
    }

    const MotionStats stats = summarizeMotion(relations);
    if (!stats.valid) {
        std::cout << "[VisualGroup] not enough reliable visual relations, fallback to single strip" << std::endl;
        return {{images, image_ids}};
    }

    std::cout << "[VisualGroup] dominant axis="
            << (stats.dominant_horizontal ? "horizontal" : "vertical")
            << ", median_main=" << stats.median_main
            << ", median_cross=" << stats.median_cross
            << ", stable_min_main=" << stats.stable_min_main
            << ", stable_max_cross=" << stats.stable_max_cross
            << std::endl;

    std::vector<VisualStripGroup> groups;
    VisualStripGroup current_group;
    current_group.images.push_back(images.front());
    current_group.image_ids.push_back(image_ids.front());

    for (size_t i = 0; i < relations.size(); ++i) {
        const auto &relation = relations[i];
        const std::string &next_id = image_ids[i + 1];

        if (isDuplicateFrame(relation, stats)) {
            std::cout << "[VisualGroup] skip near-duplicate frame: " << next_id << std::endl;
            continue;
        }

        const bool stable_step = isStableStripStep(relation, stats);
        if (!stable_step && current_group.images.size() >= 3) {
            std::cout << "[VisualGroup] strip break before: " << next_id << std::endl;
            groups.push_back(std::move(current_group));
            current_group = {};
        }

        current_group.images.push_back(images[i + 1]);
        current_group.image_ids.push_back(next_id);
    }

    if (!current_group.images.empty()) {
        groups.push_back(std::move(current_group));
    }

    for (size_t i = 0; groups.size() > 1 && i < groups.size();) {
        if (groups[i].images.size() >= 3) {
            ++i;
            continue;
        }

        if (i == 0) {
            std::cout << "[VisualGroup] merge small strip 0 into next strip" << std::endl;
            prependGroup(groups[1], groups[0]);
            groups.erase(groups.begin());
            continue;
        }

        if (i + 1 >= groups.size()) {
            std::cout << "[VisualGroup] merge trailing small strip " << i
                    << " into previous strip" << std::endl;
            appendGroup(groups[i - 1], groups[i]);
            groups.erase(groups.begin() + static_cast<long>(i));
            continue;
        }

        const auto prev_relation = estimateRelation(
            groups[i - 1].images.back(),
            groups[i].images.front(),
            tuning);
        const auto next_relation = estimateRelation(
            groups[i].images.back(),
            groups[i + 1].images.front(),
            tuning);
        const double prev_score = boundaryMergeScore(prev_relation, stats);
        const double next_score = boundaryMergeScore(next_relation, stats);

        std::cout << "[VisualGroup] small strip " << i
                << " boundary scores: prev=" << prev_score
                << " (matches=" << prev_relation.good_matches
                << ", inliers=" << prev_relation.inliers
                << ", ok=" << (prev_relation.ok ? "yes" : "no") << ")"
                << ", next=" << next_score
                << " (matches=" << next_relation.good_matches
                << ", inliers=" << next_relation.inliers
                << ", ok=" << (next_relation.ok ? "yes" : "no") << ")"
                << std::endl;

        if (next_score > prev_score) {
            std::cout << "[VisualGroup] merge small strip " << i
                    << " into next strip" << std::endl;
            prependGroup(groups[i + 1], groups[i]);
        } else {
            std::cout << "[VisualGroup] merge small strip " << i
                    << " into previous strip" << std::endl;
            appendGroup(groups[i - 1], groups[i]);
        }
        groups.erase(groups.begin() + static_cast<long>(i));
    }

    std::cout << "[VisualGroup] final strip count: " << groups.size() << std::endl;
    for (size_t i = 0; i < groups.size(); ++i) {
        std::cout << "[VisualGroup]   strip " << i
                << ": " << groups[i].images.size() << " images" << std::endl;
    }

    return groups;
}
