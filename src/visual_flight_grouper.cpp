#include "visual_flight_grouper.hpp"

#include <opencv2/features2d.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {
    struct VisualRelation {
        size_t left_index = 0;
        size_t right_index = 0;
        int gap = 1;
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
        double median_pair_score = 0.0;
        double stable_min_main = 0.0;
        double stable_max_cross = 0.0;
        double duplicate_max_main = 0.0;
        double duplicate_max_cross = 0.0;
    };

    constexpr size_t kMaxNeighborGap = 3;
    constexpr size_t kMinSegmentImages = 2;

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

    double gapWeight(const int gap) {
        return 1.0 / std::sqrt(static_cast<double>(std::max(1, gap)));
    }

    VisualRelation estimateRelation(
        const cv::Mat &left,
        const cv::Mat &right,
        const StitchTuning &tuning) {
        VisualRelation relation;
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

    const VisualRelation *findRelation(
        const std::vector<std::vector<VisualRelation> > &graph,
        const size_t left,
        const size_t right) {
        if (left >= graph.size()) {
            return nullptr;
        }
        for (const auto &relation: graph[left]) {
            if (relation.right_index == right) {
                return &relation;
            }
        }
        return nullptr;
    }

    MotionStats summarizeMotion(const std::vector<std::vector<VisualRelation> > &graph) {
        MotionStats stats;
        std::vector<double> abs_tx;
        std::vector<double> abs_ty;
        std::vector<double> pair_scores;

        for (const auto &edges: graph) {
            for (const auto &relation: edges) {
                if (!relation.ok) {
                    continue;
                }
                abs_tx.push_back(std::abs(relation.tx) / static_cast<double>(relation.gap));
                abs_ty.push_back(std::abs(relation.ty) / static_cast<double>(relation.gap));
                pair_scores.push_back(relation.score * gapWeight(relation.gap));
            }
        }

        if (abs_tx.size() < 2) {
            return stats;
        }

        stats.valid = true;
        stats.dominant_horizontal = medianOf(abs_tx) >= medianOf(abs_ty);
        stats.median_pair_score = medianOf(pair_scores);

        if (stats.dominant_horizontal) {
            stats.median_main = medianOf(abs_tx);
            stats.median_cross = medianOf(abs_ty);
        } else {
            stats.median_main = medianOf(abs_ty);
            stats.median_cross = medianOf(abs_tx);
        }

        stats.stable_min_main = std::max(18.0, stats.median_main * 0.40);
        stats.stable_max_cross = std::max(35.0, stats.median_cross * 2.60 + 8.0);
        stats.duplicate_max_main = std::max(8.0, stats.median_main * 0.12);
        stats.duplicate_max_cross = std::max(8.0, stats.median_cross * 1.50 + 4.0);
        return stats;
    }

    double mainMotion(const VisualRelation &relation, const MotionStats &stats) {
        return stats.dominant_horizontal ? relation.tx : relation.ty;
    }

    double crossMotion(const VisualRelation &relation, const MotionStats &stats) {
        return stats.dominant_horizontal ? relation.ty : relation.tx;
    }

    double mainMotionPerStep(const VisualRelation &relation, const MotionStats &stats) {
        return mainMotion(relation, stats) / static_cast<double>(std::max(1, relation.gap));
    }

    double crossMotionPerStep(const VisualRelation &relation, const MotionStats &stats) {
        return crossMotion(relation, stats) / static_cast<double>(std::max(1, relation.gap));
    }

    bool isDuplicateFrame(const VisualRelation &relation, const MotionStats &stats) {
        if (!relation.ok || relation.gap != 1) {
            return false;
        }
        return std::abs(mainMotionPerStep(relation, stats)) <= stats.duplicate_max_main &&
               std::abs(crossMotionPerStep(relation, stats)) <= stats.duplicate_max_cross;
    }

    bool isStableStripRelation(const VisualRelation &relation, const MotionStats &stats) {
        if (!relation.ok) {
            return false;
        }
        return std::abs(mainMotionPerStep(relation, stats)) >= stats.stable_min_main &&
               std::abs(crossMotionPerStep(relation, stats)) <= stats.stable_max_cross &&
               relation.scale >= 0.85 &&
               relation.scale <= 1.15 &&
               std::abs(relation.rotation_deg) <= 18.0;
    }

    double stableRelationBonus(const MotionStats &stats, const int gap) {
        return std::max(35.0, stats.median_pair_score * 1.40) * gapWeight(gap);
    }

    double uncertainRelationPenalty(const MotionStats &stats, const int gap) {
        return std::max(18.0, stats.median_pair_score * 0.60) * gapWeight(gap);
    }

    double failedRelationPenalty(const MotionStats &stats, const int gap) {
        const double base = gap == 1
                                ? std::max(28.0, stats.median_pair_score)
                                : std::max(12.0, stats.median_pair_score * 0.40);
        return base * gapWeight(gap);
    }

    double directionConflictPenalty(const MotionStats &stats) {
        return std::max(28.0, stats.median_pair_score * 0.80);
    }

    double cutPenalty(const MotionStats &stats) {
        return std::max(55.0, stats.median_pair_score * 1.60);
    }

    double relationSegmentSupport(const VisualRelation &relation, const MotionStats &stats) {
        if (!relation.ok) {
            return -failedRelationPenalty(stats, relation.gap);
        }

        double score = relation.score * gapWeight(relation.gap);
        if (isStableStripRelation(relation, stats)) {
            score += stableRelationBonus(stats, relation.gap);
        } else {
            score -= uncertainRelationPenalty(stats, relation.gap);
        }
        return score;
    }

    double directionVoteWeight(const VisualRelation &relation) {
        return gapWeight(relation.gap) * std::clamp(relation.inlier_ratio + 0.5, 0.5, 1.5);
    }

    std::vector<std::vector<VisualRelation> > buildShortRangeGraph(
        const std::vector<cv::Mat> &images,
        const std::vector<std::string> &image_ids,
        const StitchTuning &tuning) {
        std::vector<std::vector<VisualRelation> > graph(images.size());
        for (size_t i = 0; i < images.size(); ++i) {
            for (size_t gap = 1; gap <= kMaxNeighborGap && i + gap < images.size(); ++gap) {
                auto relation = estimateRelation(images[i], images[i + gap], tuning);
                relation.left_index = i;
                relation.right_index = i + gap;
                relation.gap = static_cast<int>(gap);
                graph[i].push_back(relation);

                std::cout << "[VisualGroup] edge " << i << "->" << (i + gap)
                        << " (" << image_ids[i] << " -> " << image_ids[i + gap] << ")"
                        << ": gap=" << gap
                        << ", ok=" << (relation.ok ? "yes" : "no")
                        << ", kp=" << relation.kp_left << "/" << relation.kp_right
                        << ", matches=" << relation.good_matches
                        << ", inliers=" << relation.inliers
                        << ", tx=" << relation.tx
                        << ", ty=" << relation.ty
                        << ", scale=" << relation.scale
                        << ", rot=" << relation.rotation_deg
                        << std::endl;
            }
        }
        return graph;
    }

    std::vector<std::vector<double> > buildSegmentScoreTable(
        const std::vector<std::vector<VisualRelation> > &graph,
        const MotionStats &stats) {
        const size_t n = graph.size();
        std::vector<std::vector<double> > segment_scores(
            n, std::vector<double>(n, -std::numeric_limits<double>::infinity()));

        for (size_t left = 0; left < n; ++left) {
            double base_score = 0.0;
            double positive_dir = 0.0;
            double negative_dir = 0.0;

            for (size_t right = left; right < n; ++right) {
                const size_t start = right > kMaxNeighborGap ? right - kMaxNeighborGap : 0;
                for (size_t edge_left = std::max(left, start); edge_left < right; ++edge_left) {
                    const VisualRelation *relation = findRelation(graph, edge_left, right);
                    if (!relation) {
                        continue;
                    }

                    base_score += relationSegmentSupport(*relation, stats);
                    if (isStableStripRelation(*relation, stats)) {
                        if (mainMotion(*relation, stats) >= 0.0) {
                            positive_dir += directionVoteWeight(*relation);
                        } else {
                            negative_dir += directionVoteWeight(*relation);
                        }
                    }
                }

                const size_t segment_len = right - left + 1;
                if (segment_len < kMinSegmentImages) {
                    continue;
                }

                double score = base_score;
                score -= directionConflictPenalty(stats) * std::min(positive_dir, negative_dir);
                segment_scores[left][right] = score;
            }
        }

        return segment_scores;
    }

    std::vector<std::pair<size_t, size_t> > solveBestSegmentation(
        const std::vector<std::vector<double> > &segment_scores,
        const MotionStats &stats) {
        const size_t n = segment_scores.size();
        const double neg_inf = -std::numeric_limits<double>::infinity();
        std::vector<double> best(n + 1, neg_inf);
        std::vector<int> prev(n + 1, -1);
        best[0] = 0.0;

        for (size_t end = 0; end < n; ++end) {
            for (size_t start = 0; start <= end; ++start) {
                if (end - start + 1 < kMinSegmentImages) {
                    continue;
                }
                if (!std::isfinite(segment_scores[start][end]) || !std::isfinite(best[start])) {
                    continue;
                }

                double candidate = best[start] + segment_scores[start][end];
                if (start > 0) {
                    candidate -= cutPenalty(stats);
                }

                if (candidate > best[end + 1]) {
                    best[end + 1] = candidate;
                    prev[end + 1] = static_cast<int>(start);
                }
            }
        }

        if (prev[n] < 0) {
            return {};
        }

        std::vector<std::pair<size_t, size_t> > segments;
        for (size_t cursor = n; cursor > 0;) {
            const int start = prev[cursor];
            if (start < 0) {
                return {};
            }
            segments.emplace_back(static_cast<size_t>(start), cursor - 1);
            cursor = static_cast<size_t>(start);
        }

        std::reverse(segments.begin(), segments.end());
        return segments;
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

    std::cout << "[VisualGroup] building short-range graph: images=" << images.size()
            << ", neighbor_gap=" << kMaxNeighborGap << std::endl;

    const auto graph = buildShortRangeGraph(images, image_ids, tuning);
    const MotionStats stats = summarizeMotion(graph);
    if (!stats.valid) {
        std::cout << "[VisualGroup] not enough reliable visual relations, fallback to single strip" << std::endl;
        return {{images, image_ids}};
    }

    std::cout << "[VisualGroup] dominant axis="
            << (stats.dominant_horizontal ? "horizontal" : "vertical")
            << ", median_main=" << stats.median_main
            << ", median_cross=" << stats.median_cross
            << ", median_pair_score=" << stats.median_pair_score
            << ", stable_min_main=" << stats.stable_min_main
            << ", stable_max_cross=" << stats.stable_max_cross
            << std::endl;

    std::vector<cv::Mat> filtered_images;
    std::vector<std::string> filtered_ids;
    filtered_images.reserve(images.size());
    filtered_ids.reserve(image_ids.size());
    filtered_images.push_back(images.front());
    filtered_ids.push_back(image_ids.front());
    for (size_t i = 0; i + 1 < images.size(); ++i) {
        const VisualRelation *adjacent = findRelation(graph, i, i + 1);
        if (adjacent && isDuplicateFrame(*adjacent, stats)) {
            std::cout << "[VisualGroup] remove near-duplicate frame: " << image_ids[i + 1] << std::endl;
            continue;
        }
        filtered_images.push_back(images[i + 1]);
        filtered_ids.push_back(image_ids[i + 1]);
    }
    if (filtered_images.size() < images.size()) {
        std::cout << "[VisualGroup] rerun grouping after duplicate filtering: "
                << images.size() << " -> " << filtered_images.size() << " images" << std::endl;
        return groupBoustrophedon(filtered_images, filtered_ids, tuning);
    }

    const auto segment_scores = buildSegmentScoreTable(graph, stats);
    const auto segments = solveBestSegmentation(segment_scores, stats);
    if (segments.empty()) {
        std::cout << "[VisualGroup] segmentation failed, fallback to single strip" << std::endl;
        return {{images, image_ids}};
    }

    std::vector<VisualStripGroup> groups;
    groups.reserve(segments.size());
    for (size_t gi = 0; gi < segments.size(); ++gi) {
        const auto [begin, end] = segments[gi];
        VisualStripGroup group;
        for (size_t i = begin; i <= end; ++i) {
            group.images.push_back(images[i]);
            group.image_ids.push_back(image_ids[i]);
        }
        std::cout << "[VisualGroup] segment " << gi
                << ": [" << begin << ", " << end << "]"
                << ", images=" << group.images.size()
                << ", ids=" << group.image_ids.front()
                << " -> " << group.image_ids.back()
                << std::endl;
        groups.push_back(std::move(group));
    }

    std::cout << "[VisualGroup] final strip count: " << groups.size() << std::endl;
    for (size_t i = 0; i < groups.size(); ++i) {
        std::cout << "[VisualGroup]   strip " << i
                << ": " << groups[i].images.size() << " images" << std::endl;
    }

    return groups;
}
