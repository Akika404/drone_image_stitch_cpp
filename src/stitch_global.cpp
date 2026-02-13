#include "stitch_global.hpp"

#include <opencv2/stitching/detail/blenders.hpp>
#include <opencv2/stitching/detail/exposure_compensate.hpp>
#include <opencv2/stitching/detail/matchers.hpp>
#include <opencv2/stitching/detail/motion_estimators.hpp>
#include <opencv2/stitching/detail/seam_finders.hpp>
#include <opencv2/stitching/warpers.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <utility>

#include "stitch_common.hpp"

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
