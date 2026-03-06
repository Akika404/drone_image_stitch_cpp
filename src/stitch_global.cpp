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
}

void orderStripGroupsByCrossTrack(std::vector<FlightStripGroup> &groups) {
    if (groups.size() < 2) {
        return;
    }
    std::vector<PosRecord> all_records;
    for (const auto &g: groups) {
        all_records.insert(all_records.end(), g.records.begin(), g.records.end());
    }
    const double axis_deg = averageFlightAxisHeading(all_records);
    const double axis_rad = degreeToRadian(axis_deg);
    const auto [base_lon, base_lat] = stripCentroidLonLat(groups.front().records);
    const double lat_rad = degreeToRadian(base_lat);
    constexpr double m_per_lat = 110540.0;
    const double m_per_lon = 111320.0 * std::cos(lat_rad);
    const double cross_e = -std::cos(axis_rad);
    const double cross_n = std::sin(axis_rad);

    std::vector<std::pair<double, int> > order;
    for (int i = 0; i < static_cast<int>(groups.size()); i++) {
        const auto [lon, lat] = stripCentroidLonLat(groups[i].records);
        const double e = (lon - base_lon) * m_per_lon;
        const double n = (lat - base_lat) * m_per_lat;
        order.emplace_back(e * cross_e + n * cross_n, i);
    }
    std::ranges::sort(order);
    std::vector<FlightStripGroup> sorted;
    sorted.reserve(groups.size());
    for (auto &[ct, idx]: order) {
        (void) ct;
        sorted.push_back(std::move(groups[idx]));
    }
    groups = std::move(sorted);
}

GlobalStitchInput buildGlobalStitchInput(
    const std::vector<FlightStripGroup> &strip_groups,
    const StitchTuning &tuning) {
    double base_lat = 0.0;
    double base_lon = 0.0;
    for (const auto &g: strip_groups) {
        if (!g.records.empty()) {
            base_lat = g.records[0].latitude;
            base_lon = g.records[0].longitude;
            break;
        }
    }
    const double lat_rad = degreeToRadian(base_lat);
    constexpr double m_per_lat = 110540.0;
    const double m_per_lon = 111320.0 * std::cos(lat_rad);

    struct ImgMeta {
        int flat;
        int strip;
        int pos;
        double x;
        double y;
    };
    std::vector<ImgMeta> meta;
    std::vector<cv::Mat> all_images;

    for (int s = 0; s < static_cast<int>(strip_groups.size()); s++) {
        for (int j = 0; j < static_cast<int>(strip_groups[s].images.size()); j++) {
            ImgMeta m{};
            m.flat = static_cast<int>(all_images.size());
            m.strip = s;
            m.pos = j;
            m.x = 0.0;
            m.y = 0.0;
            if (j < static_cast<int>(strip_groups[s].records.size())) {
                m.x = (strip_groups[s].records[j].longitude - base_lon) * m_per_lon;
                m.y = (strip_groups[s].records[j].latitude - base_lat) * m_per_lat;
            }
            meta.push_back(m);
            all_images.push_back(strip_groups[s].images[j]);
        }
    }

    const int n = static_cast<int>(all_images.size());
    cv::Mat mask(n, n, CV_8U, cv::Scalar(0));
    int within_pairs = 0;
    int cross_pairs = 0;

    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            if (meta[i].strip == meta[j].strip &&
                std::abs(meta[i].pos - meta[j].pos) <= tuning.range_width) {
                mask.at<uchar>(i, j) = 1;
                mask.at<uchar>(j, i) = 1;
                within_pairs++;
            }
        }
    }

    constexpr int cross_k = 4;
    const int num_strips = static_cast<int>(strip_groups.size());
    for (int s1 = 0; s1 < num_strips; s1++) {
        for (int s2 = s1 + 1; s2 <= std::min(s1 + 2, num_strips - 1); s2++) {
            for (const auto &m1: meta) {
                if (m1.strip != s1) {
                    continue;
                }
                std::vector<std::pair<double, int> > dists;
                for (const auto &m2: meta) {
                    if (m2.strip != s2) {
                        continue;
                    }
                    const double dx = m1.x - m2.x;
                    const double dy = m1.y - m2.y;
                    dists.push_back({dx * dx + dy * dy, m2.flat});
                }
                std::ranges::sort(dists);
                const int k = std::min(cross_k, static_cast<int>(dists.size()));
                for (int ki = 0; ki < k; ki++) {
                    const int j = dists[ki].second;
                    if (!mask.at<uchar>(m1.flat, j)) {
                        mask.at<uchar>(m1.flat, j) = 1;
                        mask.at<uchar>(j, m1.flat) = 1;
                        cross_pairs++;
                    }
                }
            }
        }
    }

    GlobalStitchInput result;
    result.images = std::move(all_images);
    mask.copyTo(result.match_mask);
    result.num_within_pairs = within_pairs;
    result.num_cross_pairs = cross_pairs;
    return result;
}

cv::Mat stitchGlobalPipeline(
    const std::vector<cv::Mat> &images,
    const cv::UMat &match_mask,
    const StitchTuning &tuning) {
    const std::string stage = "Global";
    const int N = static_cast<int>(images.size());
    if (N < 2) {
        throw std::runtime_error("[" + stage + "] need >= 2 images");
    }

    const double full_area = static_cast<double>(images[0].size().area());
    const double work_scale = std::min(1.0, std::sqrt(tuning.registration_resol_mpx * 1e6 / full_area));
    const double seam_scale = std::min(1.0, std::sqrt(tuning.seam_estimation_resol_mpx * 1e6 / full_area));
    const double compose_scale = (tuning.compositing_resol_mpx > 0)
                                     ? std::min(1.0, std::sqrt(tuning.compositing_resol_mpx * 1e6 / full_area))
                                     : 1.0;
    const double seam_work_aspect = seam_scale / work_scale;
    const double compose_work_aspect = compose_scale / work_scale;
    std::cout << "[" << stage << "] scale: work=" << work_scale
            << " seam=" << seam_scale << " compose=" << compose_scale << std::endl;

    std::cout << "[" << stage << "] detecting features (" << N << " images)..." << std::endl;
    cv::Ptr<cv::Feature2D> finder = cv::SIFT::create(tuning.sift_features);
    std::vector<cv::detail::ImageFeatures> features(N);
    std::vector<cv::Size> full_sizes(N);
    for (int i = 0; i < N; i++) {
        full_sizes[i] = images[i].size();
        cv::Mat work_img;
        cv::resize(images[i], work_img, cv::Size(), work_scale, work_scale, cv::INTER_LINEAR_EXACT);
        cv::detail::computeImageFeatures(finder, work_img, features[i]);
        features[i].img_idx = i;
        if ((i + 1) % 20 == 0 || i == N - 1) {
            std::cout << "[" << stage << "]   features: " << (i + 1) << "/" << N << std::endl;
        }
    }

    std::cout << "[" << stage << "] matching features (masked)..." << std::endl;
    cv::Ptr<cv::detail::FeaturesMatcher> matcher =
            cv::makePtr<cv::detail::BestOf2NearestMatcher>(tuning.try_gpu, tuning.match_conf);
    std::vector<cv::detail::MatchesInfo> pairwise_matches;
    (*matcher)(features, pairwise_matches, match_mask);
    matcher->collectGarbage();

    std::vector<int> indices = cv::detail::leaveBiggestComponent(
        features, pairwise_matches, tuning.pano_conf_thresh);
    const int num = static_cast<int>(indices.size());
    std::cout << "[" << stage << "] connected component: " << num << "/" << N << " images" << std::endl;
    if (num < 2) {
        throw std::runtime_error("[" + stage + "] only " + std::to_string(num) + " connected images");
    }

    std::vector<cv::Mat> imgs(num);
    std::vector<cv::Size> sizes_full(num);
    for (int i = 0; i < num; i++) {
        imgs[i] = images[indices[i]];
        sizes_full[i] = full_sizes[indices[i]];
    }

    std::cout << "[" << stage << "] camera estimation (affine)..." << std::endl;
    std::vector<cv::detail::CameraParams> cameras;
    if (!cv::detail::AffineBasedEstimator()(features, pairwise_matches, cameras)) {
        throw std::runtime_error("[" + stage + "] camera estimation failed");
    }
    for (auto &c: cameras) {
        c.R.convertTo(c.R, CV_32F);
    }

    std::cout << "[" << stage << "] bundle adjustment..." << std::endl;
    {
        auto ba = cv::makePtr<cv::detail::BundleAdjusterAffinePartial>();
        ba->setConfThresh(tuning.pano_conf_thresh);
        if (!(*ba)(features, pairwise_matches, cameras)) {
            std::cerr << "[" << stage << "] WARNING: bundle adjustment did not converge, "
                    << "using initial camera estimates" << std::endl;
        }
    }

    std::vector<double> focals;
    focals.reserve(cameras.size());
    for (const auto &c: cameras) {
        focals.push_back(c.focal);
    }
    std::ranges::sort(focals);
    const auto warped_scale = static_cast<float>(focals[focals.size() / 2]);

    std::cout << "[" << stage << "] warping at seam resolution (" << num << " images)..." << std::endl;
    cv::Ptr<cv::WarperCreator> wc = tuning.use_affine_warper
                                        ? static_cast<cv::Ptr<cv::WarperCreator>>(cv::makePtr<cv::AffineWarper>())
                                        : static_cast<cv::Ptr<cv::WarperCreator>>(cv::makePtr<cv::PlaneWarper>());
    auto seam_warper = wc->create(warped_scale * static_cast<float>(seam_work_aspect));

    std::vector<cv::UMat> imgs_seam(num);
    std::vector<cv::UMat> masks_seam(num);
    std::vector<cv::Point> corners_seam(num);
    std::vector<cv::Size> sizes_seam(num);

    for (int i = 0; i < num; i++) {
        cv::Mat sm;
        cv::resize(imgs[i], sm, cv::Size(), seam_scale, seam_scale, cv::INTER_LINEAR_EXACT);
        cv::Mat_<float> K;
        cameras[i].K().convertTo(K, CV_32F);
        const auto swa = static_cast<float>(seam_work_aspect);
        K(0, 0) *= swa;
        K(0, 2) *= swa;
        K(1, 1) *= swa;
        K(1, 2) *= swa;
        corners_seam[i] = seam_warper->warp(
            sm, K, cameras[i].R, cv::INTER_LINEAR, cv::BORDER_REFLECT, imgs_seam[i]);
        sizes_seam[i] = imgs_seam[i].size();
        cv::Mat m(sm.size(), CV_8U, cv::Scalar(255));
        seam_warper->warp(m, K, cameras[i].R, cv::INTER_NEAREST, cv::BORDER_CONSTANT, masks_seam[i]);
    }

    std::cout << "[" << stage << "] exposure compensation..." << std::endl;
    cv::Ptr<cv::detail::ExposureCompensator> comp = tuning.use_blocks_gain
                                                        ? cv::detail::ExposureCompensator::createDefault(
                                                            cv::detail::ExposureCompensator::GAIN_BLOCKS)
                                                        : cv::detail::ExposureCompensator::createDefault(
                                                            cv::detail::ExposureCompensator::GAIN);
    comp->feed(corners_seam, imgs_seam, masks_seam);

    std::cout << "[" << stage << "] seam finding..." << std::endl;
    {
        std::vector<cv::UMat> imgs_f(num);
        for (int i = 0; i < num; i++) {
            imgs_seam[i].convertTo(imgs_f[i], CV_32F);
        }
        auto sf = cv::makePtr<cv::detail::DpSeamFinder>(cv::detail::DpSeamFinder::COLOR_GRAD);
        sf->find(imgs_f, corners_seam, masks_seam);
    }
    imgs_seam.clear();

    std::cout << "[" << stage << "] compositing at full resolution (" << num << " images)..." << std::endl;
    auto comp_warper = wc->create(warped_scale * static_cast<float>(compose_work_aspect));

    std::vector<cv::Point> corners_c(num);
    std::vector<cv::Size> sizes_c(num);
    for (int i = 0; i < num; i++) {
        cv::Mat_<float> K;
        cameras[i].K().convertTo(K, CV_32F);
        const auto cwa = static_cast<float>(compose_work_aspect);
        K(0, 0) *= cwa;
        K(0, 2) *= cwa;
        K(1, 1) *= cwa;
        K(1, 2) *= cwa;
        cv::Size sz = sizes_full[i];
        if (std::abs(compose_scale - 1.0) > 0.1) {
            sz.width = cvRound(sz.width * compose_scale);
            sz.height = cvRound(sz.height * compose_scale);
        }
        const cv::Rect roi = comp_warper->warpRoi(sz, K, cameras[i].R);
        corners_c[i] = roi.tl();
        sizes_c[i] = roi.size();
    }

    cv::Ptr<cv::detail::Blender> blender = cv::makePtr<
        cv::detail::MultiBandBlender>(tuning.try_gpu, tuning.blend_bands);
    blender->prepare(corners_c, sizes_c);

    for (int i = 0; i < num; i++) {
        cv::Mat ci;
        if (std::abs(compose_scale - 1.0) > 0.1) {
            cv::resize(imgs[i], ci, cv::Size(), compose_scale, compose_scale, cv::INTER_LINEAR_EXACT);
        } else {
            ci = imgs[i];
        }

        cv::Mat_<float> K;
        cameras[i].K().convertTo(K, CV_32F);
        const auto cwa = static_cast<float>(compose_work_aspect);
        K(0, 0) *= cwa;
        K(0, 2) *= cwa;
        K(1, 1) *= cwa;
        K(1, 2) *= cwa;

        cv::UMat iw;
        comp_warper->warp(ci, K, cameras[i].R, cv::INTER_LINEAR, cv::BORDER_REFLECT, iw);

        cv::UMat mw;
        {
            cv::Mat fm(ci.size(), CV_8U, cv::Scalar(255));
            comp_warper->warp(fm, K, cameras[i].R, cv::INTER_NEAREST, cv::BORDER_CONSTANT, mw);
        }

        comp->apply(i, corners_c[i], iw, mw);

        cv::UMat iw_s;
        iw.convertTo(iw_s, CV_16S);
        iw.release();

        cv::UMat sd;
        cv::UMat sr;
        cv::dilate(masks_seam[i], sd, cv::Mat());
        cv::resize(sd, sr, mw.size(), 0, 0, cv::INTER_LINEAR_EXACT);

        cv::UMat cm;
        cv::bitwise_and(sr, mw, cm);

        blender->feed(iw_s, cm, corners_c[i]);

        if ((i + 1) % 10 == 0 || i == num - 1) {
            std::cout << "[" << stage << "]   composed " << (i + 1) << "/" << num << std::endl;
        }
    }

    std::cout << "[" << stage << "] blending..." << std::endl;
    cv::Mat result;
    cv::Mat result_mask;
    blender->blend(result, result_mask);
    result.convertTo(result, CV_8U);
    std::cout << "[" << stage << "] panorama: " << result.cols << "x" << result.rows << std::endl;
    return result;
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
        cv::Mat mask_src(oriented[i].size(), CV_8U, cv::Scalar(255));
        cv::Mat mask_warped;
        cv::warpAffine(
            mask_src, mask_warped, affine, sizes[i], cv::INTER_NEAREST, cv::BORDER_CONSTANT);

        warped.copyTo(warped_imgs[i]);
        mask_warped.copyTo(warped_masks[i]);
        std::cout << "[" << stage << "]   warped " << (i + 1) << "/" << num_strips << std::endl;
    }

    const double canvas_area_mpx = static_cast<double>(canvas_w) * static_cast<double>(canvas_h) / 1e6;
    const bool disable_opencl_for_large_canvas = canvas_area_mpx >= 120.0 && cv::ocl::useOpenCL();
    if (disable_opencl_for_large_canvas) {
        std::cout << "[" << stage << "] canvas is large (" << canvas_area_mpx
                << " MP), disabling OpenCL for custom global compose" << std::endl;
        cv::ocl::setUseOpenCL(false);
    }

    auto make_exposure_compensator = [&](const bool prefer_blocks, std::string &mode_out) {
        cv::Ptr<cv::detail::ExposureCompensator> local_compensator;
        if (prefer_blocks && canvas_area_mpx <= 120.0) {
            const int block_size = canvas_area_mpx > 80.0 ? 96 : (canvas_area_mpx > 45.0 ? 64 : 32);
            auto blocks_channels = cv::makePtr<cv::detail::BlocksChannelsCompensator>(block_size, block_size, 2);
            blocks_channels->setNrGainsFilteringIterations(canvas_area_mpx > 45.0 ? 1 : 2);
            blocks_channels->setSimilarityThreshold(0.85);
            local_compensator = blocks_channels;
            mode_out = "CHANNELS_BLOCKS";
        } else if (canvas_area_mpx <= 180.0) {
            auto channels = cv::makePtr<cv::detail::ChannelsCompensator>(2);
            channels->setSimilarityThreshold(0.9);
            local_compensator = channels;
            mode_out = "CHANNELS";
        } else {
            auto gain = cv::makePtr<cv::detail::GainCompensator>(1);
            gain->setSimilarityThreshold(0.9);
            local_compensator = gain;
            mode_out = "GAIN";
        }
        return local_compensator;
    };

    std::string exposure_mode;
    cv::Ptr<cv::detail::ExposureCompensator> compensator =
        make_exposure_compensator(tuning.use_blocks_gain, exposure_mode);
    std::cout << "[" << stage << "] exposure compensation begin, mode=" << exposure_mode
            << ", canvas_mpx=" << canvas_area_mpx << std::endl;
    try {
        compensator->feed(corners, warped_imgs, warped_masks);
    } catch (const cv::Exception &e) {
        if (!looksLikeOpenClFailure(e) || disable_opencl_for_large_canvas) {
            throw;
        }
        std::cerr << "[" << stage << "] exposure compensation OpenCL failure, retry on CPU: "
                << e.what() << std::endl;
        cv::ocl::setUseOpenCL(false);
        compensator = make_exposure_compensator(false, exposure_mode);
        std::cout << "[" << stage << "] exposure compensation retry, mode=" << exposure_mode << std::endl;
        compensator->feed(corners, warped_imgs, warped_masks);
    }
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
        }
    } else {
        for (int i = 0; i < num_strips; ++i) {
            seam_masks[i] = warped_masks[i].clone();
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

    if (seam_scale < 0.999) {
        for (int i = 0; i < num_strips; ++i) {
            cv::UMat seam_mask_full;
            cv::resize(seam_masks[i], seam_mask_full, warped_masks[i].size(), 0, 0, cv::INTER_NEAREST);
            cv::bitwise_and(seam_mask_full, warped_masks[i], seam_masks[i]);
        }
    } else {
        for (int i = 0; i < num_strips; ++i) {
            cv::bitwise_and(seam_masks[i], warped_masks[i], seam_masks[i]);
        }
    }
    std::cout << "[" << stage << "] seam finding done" << std::endl;

    cv::Ptr<cv::detail::Blender> blender = cv::makePtr<cv::detail::MultiBandBlender>(
        tuning.try_gpu, std::max(1, tuning.blend_bands));
    blender->prepare(corners, sizes);
    std::cout << "[" << stage << "] blender prepared, blend_bands="
            << std::max(1, tuning.blend_bands) << std::endl;

    for (int i = 0; i < num_strips; ++i) {
        compensator->apply(i, corners[i], warped_imgs[i], warped_masks[i]);

        cv::UMat warped_16s;
        warped_imgs[i].convertTo(warped_16s, CV_16S);

        cv::UMat seam_dilated;
        cv::dilate(seam_masks[i], seam_dilated, cv::Mat());
        cv::UMat blend_mask;
        cv::bitwise_and(seam_dilated, warped_masks[i], blend_mask);
        cv::GaussianBlur(blend_mask, blend_mask, cv::Size(31, 31), 0.0, 0.0, cv::BORDER_DEFAULT);
        cv::bitwise_and(blend_mask, warped_masks[i], blend_mask);
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
    return result;
}
