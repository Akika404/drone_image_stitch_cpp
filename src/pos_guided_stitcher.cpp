#include "pos_guided_stitcher.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/features2d.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/stitching.hpp>
#include <opencv2/stitching/detail/blenders.hpp>
#include <opencv2/stitching/detail/camera.hpp>
#include <opencv2/stitching/detail/exposure_compensate.hpp>
#include <opencv2/stitching/detail/matchers.hpp>
#include <opencv2/stitching/detail/motion_estimators.hpp>
#include <opencv2/stitching/detail/seam_finders.hpp>
#include <opencv2/stitching/detail/util.hpp>
#include <opencv2/stitching/detail/warpers.hpp>

namespace fs = std::filesystem;

namespace {
    constexpr double kEarthRadiusMeters = 6378137.0;

    cv::Point2d lonLatToLocal(const PosRecord &pos, const double lon0, const double lat0) {
        const double lat0_rad = lat0 * CV_PI / 180.0;
        const double x = (pos.longitude - lon0) * CV_PI / 180.0 * kEarthRadiusMeters * std::cos(lat0_rad);
        const double y = (pos.latitude - lat0) * CV_PI / 180.0 * kEarthRadiusMeters;
        return {x, y};
    }

    double median(std::vector<double> values) {
        if (values.empty()) {
            return 0.0;
        }
        const auto mid = values.begin() + static_cast<long>(values.size() / 2);
        std::nth_element(values.begin(), mid, values.end());
        if (values.size() % 2 == 1) {
            return *mid;
        }
        const auto mid_prev = mid - 1;
        return (*mid_prev + *mid) * 0.5;
    }

    cv::Point2d applyLinearMap(const cv::Matx22d &a, const cv::Point2d &v) {
        return {
            a(0, 0) * v.x + a(0, 1) * v.y,
            a(1, 0) * v.x + a(1, 1) * v.y
        };
    }

    bool fitPosToImageLinearMap(const std::vector<cv::Point2d> &predicted,
                                const std::vector<cv::Point2d> &measured,
                                cv::Matx22d &map_out) {
        if (predicted.size() < 3 || predicted.size() != measured.size()) {
            return false;
        }

        auto solveMap = [&](const std::vector<int> &idxs, cv::Matx22d &out) -> bool {
            if (idxs.size() < 3) {
                return false;
            }
            cv::Mat m(static_cast<int>(idxs.size()), 2, CV_64F);
            cv::Mat vx(static_cast<int>(idxs.size()), 1, CV_64F);
            cv::Mat vy(static_cast<int>(idxs.size()), 1, CV_64F);
            for (size_t r = 0; r < idxs.size(); ++r) {
                const cv::Point2d p = predicted[static_cast<size_t>(idxs[r])];
                const cv::Point2d q = measured[static_cast<size_t>(idxs[r])];
                m.at<double>(static_cast<int>(r), 0) = p.x;
                m.at<double>(static_cast<int>(r), 1) = p.y;
                vx.at<double>(static_cast<int>(r), 0) = q.x;
                vy.at<double>(static_cast<int>(r), 0) = q.y;
            }
            cv::Mat ax;
            cv::Mat ay;
            if (!cv::solve(m, vx, ax, cv::DECOMP_SVD)) {
                return false;
            }
            if (!cv::solve(m, vy, ay, cv::DECOMP_SVD)) {
                return false;
            }
            out = cv::Matx22d(
                ax.at<double>(0, 0), ax.at<double>(1, 0),
                ay.at<double>(0, 0), ay.at<double>(1, 0));
            return true;
        };

        std::vector<int> all_idxs(predicted.size());
        std::iota(all_idxs.begin(), all_idxs.end(), 0);
        cv::Matx22d map;
        if (!solveMap(all_idxs, map)) {
            return false;
        }

        std::vector<double> residuals;
        residuals.reserve(predicted.size());
        for (size_t i = 0; i < predicted.size(); ++i) {
            const cv::Point2d est = applyLinearMap(map, predicted[i]);
            residuals.push_back(cv::norm(est - measured[i]));
        }
        const double med_res = median(residuals);
        const double inlier_gate = std::max(5.0, med_res * 2.8);

        std::vector<int> inliers;
        inliers.reserve(predicted.size());
        for (size_t i = 0; i < residuals.size(); ++i) {
            if (residuals[i] <= inlier_gate) {
                inliers.push_back(static_cast<int>(i));
            }
        }
        if (inliers.size() >= 3) {
            cv::Matx22d refined;
            if (solveMap(inliers, refined)) {
                map = refined;
            }
        }

        const double s1 = std::sqrt(map(0, 0) * map(0, 0) + map(1, 0) * map(1, 0));
        const double s2 = std::sqrt(map(0, 1) * map(0, 1) + map(1, 1) * map(1, 1));
        const double det = std::abs(map(0, 0) * map(1, 1) - map(0, 1) * map(1, 0));
        const double anis = std::max(s1, s2) / std::max(1e-6, std::min(s1, s2));
        if (s1 < 0.35 || s1 > 3.0 || s2 < 0.35 || s2 > 3.0 || anis > 2.5 || det < 0.05) {
            return false;
        }

        map_out = map;
        return true;
    }

    double calcImageScaleMpx(const cv::Size &size, const double target_mpx) {
        if (target_mpx <= 0) {
            return 1.0;
        }
        const double area = static_cast<double>(size.area());
        if (area <= 0.0) {
            return 1.0;
        }
        return std::min(1.0, std::sqrt(target_mpx * 1e6 / area));
    }

    double normalizeAngleDeg(double angle_deg) {
        while (angle_deg <= -180.0) {
            angle_deg += 360.0;
        }
        while (angle_deg > 180.0) {
            angle_deg -= 360.0;
        }
        return angle_deg;
    }

    cv::Mat rotateImageExpand(const cv::Mat &src, const double angle_deg) {
        if (src.empty()) {
            return src;
        }
        if (std::abs(angle_deg) < 0.1) {
            return src;
        }

        const cv::Point2f center(static_cast<float>(src.cols) * 0.5f, static_cast<float>(src.rows) * 0.5f);
        cv::Mat rot = cv::getRotationMatrix2D(center, angle_deg, 1.0);

        const double abs_cos = std::abs(rot.at<double>(0, 0));
        const double abs_sin = std::abs(rot.at<double>(0, 1));
        const int bound_w = static_cast<int>(std::ceil(src.rows * abs_sin + src.cols * abs_cos));
        const int bound_h = static_cast<int>(std::ceil(src.rows * abs_cos + src.cols * abs_sin));

        rot.at<double>(0, 2) += bound_w * 0.5 - center.x;
        rot.at<double>(1, 2) += bound_h * 0.5 - center.y;

        cv::Mat dst;
        cv::warpAffine(src, dst, rot, cv::Size(bound_w, bound_h), cv::INTER_LINEAR, cv::BORDER_REFLECT);
        return dst;
    }

    bool phaseCorrelateSafe(const cv::Mat &a, const cv::Mat &b, cv::Point2d &shift, double &response) {
        if (a.empty() || b.empty()) {
            return false;
        }
        cv::Mat a32 = a;
        cv::Mat b32 = b;
        if (a32.type() != CV_32F) {
            a32.convertTo(a32, CV_32F);
        }
        if (b32.type() != CV_32F) {
            b32.convertTo(b32, CV_32F);
        }

        if (a32.size() != b32.size()) {
            const int w = std::min(a32.cols, b32.cols);
            const int h = std::min(a32.rows, b32.rows);
            if (w < 64 || h < 64) {
                return false;
            }
            const cv::Rect ra((a32.cols - w) / 2, (a32.rows - h) / 2, w, h);
            const cv::Rect rb((b32.cols - w) / 2, (b32.rows - h) / 2, w, h);
            a32 = a32(ra);
            b32 = b32(rb);
        }

        shift = cv::phaseCorrelate(a32, b32, cv::noArray(), &response);
        return true;
    }

    cv::UMat buildMatchingMask(const std::vector<cv::Point2d> &coords, const StitchTuning &tuning) {
        const int n = static_cast<int>(coords.size());
        cv::Mat mask = cv::Mat::zeros(n, n, CV_8U);
        if (n == 0) {
            cv::UMat out;
            mask.copyTo(out);
            return out;
        }

        for (int i = 0; i < n; ++i) {
            mask.at<uchar>(i, i) = 1;
        }

        std::vector<double> steps;
        steps.reserve(std::max(0, n - 1));
        for (int i = 1; i < n; ++i) {
            steps.push_back(cv::norm(coords[i] - coords[i - 1]));
        }
        const double median_step = std::max(5.0, median(steps));
        const double spatial_radius = std::max(25.0, median_step * 5.0);
        const int range_width = std::max(1, tuning.range_width);
        const int nearest_count = std::max(6, std::min(14, n - 1));

        for (int i = 0; i < n; ++i) {
            const int lo = std::max(0, i - range_width);
            const int hi = std::min(n - 1, i + range_width);
            for (int j = lo; j <= hi; ++j) {
                mask.at<uchar>(i, j) = 1;
                mask.at<uchar>(j, i) = 1;
            }

            std::vector<std::pair<double, int> > dists;
            dists.reserve(n - 1);
            for (int j = 0; j < n; ++j) {
                if (i == j) {
                    continue;
                }
                const double d = cv::norm(coords[i] - coords[j]);
                if (d <= spatial_radius) {
                    mask.at<uchar>(i, j) = 1;
                    mask.at<uchar>(j, i) = 1;
                }
                dists.emplace_back(d, j);
            }
            std::ranges::sort(dists, [](const auto &a, const auto &b) {
                return a.first < b.first;
            });
            for (int k = 0; k < nearest_count && k < static_cast<int>(dists.size()); ++k) {
                const int j = dists[k].second;
                mask.at<uchar>(i, j) = 1;
                mask.at<uchar>(j, i) = 1;
            }
        }

        cv::UMat out;
        mask.copyTo(out);
        return out;
    }

    template<class T>
    std::vector<T> subsetByIndices(const std::vector<T> &src, const std::vector<int> &indices) {
        std::vector<T> out;
        out.reserve(indices.size());
        for (const int idx: indices) {
            out.push_back(src[static_cast<size_t>(idx)]);
        }
        return out;
    }

    cv::Ptr<cv::detail::RotationWarper> makeWarper(const StitchTuning &tuning, const float scale) {
        if (tuning.use_affine_warper) {
            return cv::makePtr<cv::detail::AffineWarper>(scale);
        }
        return cv::makePtr<cv::detail::PlaneWarper>(scale);
    }

    cv::Mat buildFeatherWeight(const cv::Size &size) {
        cv::Mat weight(size, CV_32F, cv::Scalar::all(1.0f));
        const int bx = std::max(8, std::min(size.width / 6, 96));
        const int by = std::max(8, std::min(size.height / 6, 96));

        std::vector<float> wx(size.width, 1.0f);
        std::vector<float> wy(size.height, 1.0f);
        for (int x = 0; x < bx; ++x) {
            const float v = static_cast<float>(x + 1) / static_cast<float>(bx + 1);
            wx[x] = v;
            wx[size.width - 1 - x] = std::min(wx[size.width - 1 - x], v);
        }
        for (int y = 0; y < by; ++y) {
            const float v = static_cast<float>(y + 1) / static_cast<float>(by + 1);
            wy[y] = v;
            wy[size.height - 1 - y] = std::min(wy[size.height - 1 - y], v);
        }

        for (int y = 0; y < size.height; ++y) {
            float *row = weight.ptr<float>(y);
            for (int x = 0; x < size.width; ++x) {
                row[x] = wx[x] * wy[y];
            }
        }
        return weight;
    }

    std::vector<std::pair<size_t, size_t> > buildStripRanges(const std::vector<cv::Point2d> &pos_centers) {
        std::vector<std::pair<size_t, size_t> > ranges;
        const size_t n = pos_centers.size();
        if (n == 0) {
            return ranges;
        }
        if (n < 3) {
            ranges.emplace_back(0, n);
            return ranges;
        }

        cv::Point2d axis(0.0, 1.0);
        double best_len = 0.0;
        const size_t probe_end = std::min<size_t>(n, 40);
        for (size_t i = 1; i < probe_end; ++i) {
            const cv::Point2d d = pos_centers[i] - pos_centers[i - 1];
            const double len = cv::norm(d);
            if (len > best_len && len > 1.0) {
                best_len = len;
                axis = d * (1.0 / len);
            }
        }
        if (cv::norm(axis) < 1e-6) {
            axis = cv::Point2d(0.0, 1.0);
        }

        const auto signAlong = [&](const cv::Point2d &d) -> int {
            const double along = d.dot(axis);
            if (std::abs(along) < 1.0) {
                return 0;
            }
            return along > 0 ? 1 : -1;
        };

        size_t start = 0;
        int current_sign = 0;
        for (size_t i = 1; i < n; ++i) {
            const cv::Point2d d = pos_centers[i] - pos_centers[i - 1];
            const int s = signAlong(d);
            if (s == 0) {
                continue;
            }
            if (current_sign == 0) {
                current_sign = s;
                continue;
            }
            if (s != current_sign) {
                const size_t strip_len = i - start;
                if (strip_len >= 10) {
                    ranges.emplace_back(start, i);
                    start = i;
                    current_sign = 0;
                    continue;
                }
            }
            current_sign = s;
        }
        if (start < n) {
            ranges.emplace_back(start, n);
        }
        return ranges;
    }

    double estimatePixelsPerMeter(const std::vector<StitchFrame> &frames,
                                  const std::vector<cv::Point2d> &coords,
                                  const double compose_scale) {
        std::vector<double> ratios;
        ratios.reserve(std::min<size_t>(frames.size(), 80));

        const size_t max_pairs = std::min<size_t>(frames.size() - 1, 80);
        for (size_t i = 0; i < max_pairs; ++i) {
            const double dist_m = cv::norm(coords[i + 1] - coords[i]);
            if (dist_m < 0.2 || dist_m > 200.0) {
                continue;
            }

            cv::Mat img_a = cv::imread(frames[i].path, cv::IMREAD_GRAYSCALE);
            cv::Mat img_b = cv::imread(frames[i + 1].path, cv::IMREAD_GRAYSCALE);
            if (img_a.empty() || img_b.empty()) {
                continue;
            }

            const double probe_scale = std::min(0.5, compose_scale);
            if (probe_scale < 0.999) {
                cv::resize(img_a, img_a, cv::Size(), probe_scale, probe_scale, cv::INTER_AREA);
                cv::resize(img_b, img_b, cv::Size(), probe_scale, probe_scale, cv::INTER_AREA);
            }

            cv::Mat f1;
            cv::Mat f2;
            img_a.convertTo(f1, CV_32F);
            img_b.convertTo(f2, CV_32F);
            double response = 0.0;
            const cv::Point2d shift = cv::phaseCorrelate(f1, f2, cv::noArray(), &response);
            const double pix_shift = cv::norm(shift);
            if (response < 0.12 || pix_shift < 1.0) {
                continue;
            }
            ratios.push_back(pix_shift / dist_m);
        }

        if (!ratios.empty()) {
            return std::max(1.0, median(ratios));
        }

        // 无法从相邻帧估计时的保守回退值。
        return 8.0;
    }

    cv::Mat composeWinnerByCenters(const std::vector<cv::Mat> &images,
                                   const std::vector<cv::Point2d> &centers) {
        if (images.empty() || images.size() != centers.size()) {
            return {};
        }

        double min_x = std::numeric_limits<double>::max();
        double min_y = std::numeric_limits<double>::max();
        double max_x = std::numeric_limits<double>::lowest();
        double max_y = std::numeric_limits<double>::lowest();
        for (size_t i = 0; i < images.size(); ++i) {
            const auto &img = images[i];
            if (img.empty()) {
                continue;
            }
            const double tl_x = centers[i].x - img.cols * 0.5;
            const double tl_y = centers[i].y - img.rows * 0.5;
            min_x = std::min(min_x, tl_x);
            min_y = std::min(min_y, tl_y);
            max_x = std::max(max_x, tl_x + img.cols);
            max_y = std::max(max_y, tl_y + img.rows);
        }

        const int canvas_w = static_cast<int>(std::ceil(max_x - min_x));
        const int canvas_h = static_cast<int>(std::ceil(max_y - min_y));
        if (canvas_w <= 0 || canvas_h <= 0 || canvas_w > 400000 || canvas_h > 400000) {
            return {};
        }

        cv::Mat best_img(canvas_h, canvas_w, CV_8UC3, cv::Scalar::all(0));
        cv::Mat best_w(canvas_h, canvas_w, CV_32F, cv::Scalar::all(0));
        cv::Size cached_size;
        cv::Mat cached_weight;
        for (size_t i = 0; i < images.size(); ++i) {
            const auto &img = images[i];
            if (img.empty()) {
                continue;
            }
            if (cached_size != img.size()) {
                cached_weight = buildFeatherWeight(img.size());
                cached_size = img.size();
            }

            const int x = static_cast<int>(std::lround(centers[i].x - img.cols * 0.5 - min_x));
            const int y = static_cast<int>(std::lround(centers[i].y - img.rows * 0.5 - min_y));
            const cv::Rect roi(x, y, img.cols, img.rows);
            if (roi.x < 0 || roi.y < 0 || roi.br().x > canvas_w || roi.br().y > canvas_h) {
                continue;
            }

            cv::Mat roi_best_w = best_w(roi);
            cv::Mat better_mask;
            cv::compare(cached_weight, roi_best_w, better_mask, cv::CMP_GT);
            cached_weight.copyTo(roi_best_w, better_mask);
            img.copyTo(best_img(roi), better_mask);
        }
        return best_img;
    }

    StitchOutput stitchByStripMosaics(const std::vector<StitchFrame> &frames,
                                      const StitchTuning &tuning,
                                      const std::string &output_path,
                                      const std::string &reason) {
        StitchOutput output;
        output.input_count = static_cast<int>(frames.size());
        output.output_path = output_path;
        if (frames.size() < 4) {
            output.message = "strip-mosaic needs at least 4 frames";
            return output;
        }

        const double lon0 = frames.front().pos.longitude;
        const double lat0 = frames.front().pos.latitude;
        std::vector<cv::Point2d> local_coords;
        local_coords.reserve(frames.size());
        for (const auto &frame: frames) {
            local_coords.push_back(lonLatToLocal(frame.pos, lon0, lat0));
        }

        cv::Mat probe = cv::imread(frames.front().path, cv::IMREAD_COLOR);
        if (probe.empty()) {
            output.message = "failed to read probe image in strip-mosaic mode";
            return output;
        }

        double compose_scale = tuning.compositing_resol_mpx <= 0
                                   ? (frames.size() > 200 ? 0.22 : 0.40)
                                   : calcImageScaleMpx(probe.size(), tuning.compositing_resol_mpx);
        compose_scale = std::clamp(compose_scale, 0.12, 0.65);
        const double ppm = estimatePixelsPerMeter(frames, local_coords, compose_scale);

        std::vector<cv::Point2d> pos_centers(frames.size());
        for (size_t i = 0; i < frames.size(); ++i) {
            pos_centers[i] = {local_coords[i].x * ppm, -local_coords[i].y * ppm};
        }

        const auto strips = buildStripRanges(pos_centers);
        if (strips.size() < 2) {
            output.message = "strip detection failed";
            return output;
        }

        std::vector<cv::Mat> strip_panos;
        std::vector<cv::Point2d> strip_centers;
        int used_frames = 0;

        std::cout << "[STITCH] strip-mosaic mode: strips=" << strips.size()
                << ", reason=" << reason << std::endl;
        for (size_t si = 0; si < strips.size(); ++si) {
            const auto [s, e] = strips[si];
            if (e <= s || e - s < 3) {
                continue;
            }

            std::vector<size_t> candidate_indices;
            candidate_indices.reserve(e - s);
            for (size_t i = s; i < e; ++i) {
                candidate_indices.push_back(i);
            }
            if (candidate_indices.size() < 3) {
                continue;
            }

            // 航带内按“沿航向推进距离”选关键帧，去掉低位移和端点转弯重复帧。
            cv::Point2d axis = pos_centers[candidate_indices.back()] - pos_centers[candidate_indices.front()];
            double axis_norm = cv::norm(axis);
            if (axis_norm < 1e-6) {
                continue;
            }
            axis *= (1.0 / axis_norm);
            std::vector<double> step_norms;
            step_norms.reserve(candidate_indices.size() - 1);
            for (size_t i = 1; i < candidate_indices.size(); ++i) {
                step_norms.push_back(cv::norm(
                        pos_centers[candidate_indices[i]] - pos_centers[candidate_indices[i - 1]]));
            }
            const double med_step = std::max(6.0, median(step_norms));
            const double min_progress = std::max(6.0, med_step * 0.45);

            std::vector<size_t> loaded_indices;
            loaded_indices.reserve(candidate_indices.size());
            loaded_indices.push_back(candidate_indices.front());
            double last_proj = pos_centers[candidate_indices.front()].dot(axis);
            for (size_t i = 1; i < candidate_indices.size(); ++i) {
                const size_t idx = candidate_indices[i];
                const double proj = pos_centers[idx].dot(axis);
                if (std::abs(proj - last_proj) < min_progress) {
                    continue;
                }
                loaded_indices.push_back(idx);
                last_proj = proj;
            }
            if (loaded_indices.size() < 3) {
                continue;
            }

            std::vector<double> kappa_values;
            kappa_values.reserve(loaded_indices.size());
            for (const size_t idx: loaded_indices) {
                kappa_values.push_back(frames[idx].pos.kappa);
            }
            const double strip_kappa_med = median(kappa_values);

            std::vector<cv::Mat> strip_images;
            strip_images.reserve(loaded_indices.size());
            std::vector<cv::Point2d> strip_pos;
            strip_pos.reserve(loaded_indices.size());
            for (const size_t idx: loaded_indices) {
                cv::Mat img = cv::imread(frames[idx].path, cv::IMREAD_COLOR);
                if (img.empty()) {
                    continue;
                }
                if (compose_scale < 0.999) {
                    cv::resize(img, img, cv::Size(), compose_scale, compose_scale, cv::INTER_AREA);
                }

                // 以该航带 κ 中位数为参考，将每帧旋转到统一朝向，降低错位风险。
                const double yaw_delta = normalizeAngleDeg(frames[idx].pos.kappa - strip_kappa_med);
                if (std::abs(yaw_delta) > 45.0) {
                    continue;
                }
                img = rotateImageExpand(img, -yaw_delta);
                strip_images.push_back(img);
                strip_pos.push_back(pos_centers[idx]);
            }
            if (strip_images.size() < 3) {
                continue;
            }

            // 去掉几乎重叠的相邻帧，减少重复边缘。
            std::vector<cv::Mat> use_images;
            std::vector<cv::Point2d> use_pos;
            use_images.reserve(strip_images.size());
            use_pos.reserve(strip_images.size());
            use_images.push_back(strip_images.front());
            use_pos.push_back(strip_pos.front());
            for (size_t i = 1; i < strip_images.size(); ++i) {
                if (cv::norm(strip_pos[i] - use_pos.back()) < 6.0) {
                    continue;
                }
                use_images.push_back(strip_images[i]);
                use_pos.push_back(strip_pos[i]);
            }
            const size_t min_strip_frames = frames.size() > 200 ? 20 : 8;
            if (use_images.size() < min_strip_frames) {
                std::cout << "  [strip] " << si << " skipped (too short): frames="
                        << use_images.size() << std::endl;
                continue;
            }
            if (use_images.size() < 2) {
                std::cout << "  [strip] " << si << " failed" << std::endl;
                continue;
            }

            std::vector<cv::Mat> use_grays;
            use_grays.reserve(use_images.size());
            for (const auto &img: use_images) {
                cv::Mat gray;
                cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
                gray.convertTo(gray, CV_32F);
                use_grays.push_back(gray);
            }

            std::vector<cv::Point2d> pair_predicted;
            std::vector<cv::Point2d> pair_measured;
            pair_predicted.reserve(use_images.size() - 1);
            pair_measured.reserve(use_images.size() - 1);
            std::vector<char> step_has_measure(use_images.size() - 1, 0);
            std::vector<cv::Point2d> step_measure(use_images.size() - 1);
            std::vector<double> step_response(use_images.size() - 1, 0.0);
            for (size_t i = 1; i < use_images.size(); ++i) {
                const cv::Point2d predicted = use_pos[i] - use_pos[i - 1];
                double response = 0.0;
                cv::Point2d shift;
                if (!phaseCorrelateSafe(use_grays[i - 1], use_grays[i], shift, response)) {
                    continue;
                }
                if (response < 0.18) {
                    continue;
                }
                const cv::Point2d measured(-shift.x, -shift.y);
                const double pred_norm = std::max(1e-3, cv::norm(predicted));
                const double meas_norm = cv::norm(measured);
                const double ratio = meas_norm / pred_norm;
                if (ratio < 0.20 || ratio > 5.0) {
                    continue;
                }

                pair_predicted.push_back(predicted);
                pair_measured.push_back(measured);
                step_has_measure[i - 1] = 1;
                step_measure[i - 1] = measured;
                step_response[i - 1] = response;
            }

            cv::Matx22d pos_to_img_map = cv::Matx22d::eye();
            const bool map_ok = fitPosToImageLinearMap(pair_predicted, pair_measured, pos_to_img_map);

            std::vector<cv::Point2d> use_centers(use_images.size());
            bool base_from_map = false;
            if (map_ok) {
                const cv::Point2d anchor = use_pos.front();
                const cv::Point2d pos_span = use_pos.back() - anchor;
                const cv::Point2d map_span = applyLinearMap(pos_to_img_map, pos_span);
                const double span_ratio = cv::norm(map_span) / std::max(1e-3, cv::norm(pos_span));
                if (span_ratio < 0.65 || span_ratio > 1.60) {
                    use_centers[0] = use_pos[0];
                    for (size_t i = 1; i < use_images.size(); ++i) {
                        const cv::Point2d predicted = use_pos[i] - use_pos[i - 1];
                        cv::Point2d fused = predicted;
                        if (step_has_measure[i - 1]) {
                            fused = predicted * 0.85 + step_measure[i - 1] * 0.15;
                        }
                        use_centers[i] = use_centers[i - 1] + fused;
                    }
                } else {
                    for (size_t i = 0; i < use_images.size(); ++i) {
                        const cv::Point2d local = use_pos[i] - anchor;
                        use_centers[i] = anchor + applyLinearMap(pos_to_img_map, local);
                    }
                    base_from_map = true;
                }
            } else {
                use_centers[0] = use_pos[0];
                for (size_t i = 0; i < use_images.size(); ++i) {
                    if (i == 0) {
                        continue;
                    }
                    const cv::Point2d predicted = use_pos[i] - use_pos[i - 1];
                    cv::Point2d fused = predicted;
                    if (step_has_measure[i - 1]) {
                        fused = predicted * 0.85 + step_measure[i - 1] * 0.15;
                    }
                    use_centers[i] = use_centers[i - 1] + fused;
                }
            }

            if (!base_from_map) {
                std::vector<cv::Point2d> refined_centers(use_centers.size());
                refined_centers[0] = use_centers[0];
                for (size_t i = 1; i < use_centers.size(); ++i) {
                    const cv::Point2d prior = use_centers[i] - use_centers[i - 1];
                    cv::Point2d fused = prior;
                    if (step_has_measure[i - 1]) {
                        const cv::Point2d measured = step_measure[i - 1];
                        const double diff = cv::norm(measured - prior);
                        const double gate = std::max(5.0, cv::norm(prior) * 0.45 + 5.0);
                        if (diff <= gate) {
                            const double conf = std::clamp((step_response[i - 1] - 0.18) / 0.42, 0.0, 1.0);
                            const double alpha = 0.10 + 0.25 * conf;
                            fused = prior * (1.0 - alpha) + measured * alpha;
                        }
                    }
                    refined_centers[i] = refined_centers[i - 1] + fused;
                }

                const cv::Point2d err_start = use_pos.front() - refined_centers.front();
                const cv::Point2d err_end = use_pos.back() - refined_centers.back();
                const double denom = static_cast<double>(std::max<size_t>(1, refined_centers.size() - 1));
                for (size_t i = 0; i < refined_centers.size(); ++i) {
                    const double t = static_cast<double>(i) / denom;
                    refined_centers[i] += (err_start * (1.0 - t) + err_end * t) * 0.35;
                }
                use_centers.swap(refined_centers);
            }

            cv::Mat pano = composeWinnerByCenters(use_images, use_centers);
            if (pano.empty()) {
                std::cout << "  [strip] " << si << " failed" << std::endl;
                continue;
            }

            cv::Point2d center(0.0, 0.0);
            int count = 0;
            for (const auto &p: use_pos) {
                center += p;
                ++count;
            }
            center *= (1.0 / std::max(1, count));

            strip_panos.push_back(pano);
            strip_centers.push_back(center);
            used_frames += static_cast<int>(use_images.size());
            std::cout << "  [strip] " << si << " ok, frames=" << use_images.size()
                    << ", pair_fit=" << pair_measured.size()
                    << ", map=" << (map_ok ? "ok" : "phase-chain")
                    << ", pano=" << pano.cols << "x" << pano.rows << std::endl;
        }

        if (strip_panos.size() < 2) {
            output.message = "not enough strip panoramas";
            return output;
        }

        cv::Mat final_img = composeWinnerByCenters(strip_panos, strip_centers);
        if (final_img.empty()) {
            output.message = "failed to compose strip panoramas";
            return output;
        }

        fs::path out_path(output_path);
        fs::create_directories(out_path.parent_path());
        if (!cv::imwrite(output_path, final_img)) {
            output.message = "failed to write output image: " + output_path;
            return output;
        }

        output.ok = true;
        output.stitched_count = used_frames;
        output.message = "stitch completed (strip-mosaic fallback)";
        return output;
    }

    StitchOutput stitchByPosProjection(const std::vector<StitchFrame> &frames,
                                       const StitchTuning &tuning,
                                       const std::string &output_path,
                                       const std::string &reason) {
        StitchOutput output;
        output.input_count = static_cast<int>(frames.size());
        output.stitched_count = static_cast<int>(frames.size());
        output.output_path = output_path;
        if (frames.size() < 2) {
            output.message = "need at least 2 frames after POS alignment";
            return output;
        }

        const double lon0 = frames.front().pos.longitude;
        const double lat0 = frames.front().pos.latitude;
        std::vector<cv::Point2d> local_coords;
        local_coords.reserve(frames.size());
        for (const auto &frame: frames) {
            local_coords.push_back(lonLatToLocal(frame.pos, lon0, lat0));
        }

        cv::Mat probe = cv::imread(frames.front().path, cv::IMREAD_COLOR);
        if (probe.empty()) {
            output.message = "failed to read first image for POS-projection mode";
            return output;
        }

        double compose_scale = tuning.compositing_resol_mpx <= 0
                                   ? (frames.size() > 200 ? 0.25 : 0.45)
                                   : calcImageScaleMpx(probe.size(), tuning.compositing_resol_mpx);
        if (frames.size() > 200) {
            compose_scale = std::min(compose_scale, 0.35);
        }
        compose_scale = std::clamp(compose_scale, 0.12, 1.0);
        const double pixels_per_meter = estimatePixelsPerMeter(frames, local_coords, compose_scale);

        std::vector<cv::Mat> images;
        images.reserve(frames.size());
        std::vector<cv::Mat> grays;
        grays.reserve(frames.size());
        std::vector<cv::Point2d> centers(frames.size());
        std::vector<cv::Point2d> pos_centers(frames.size());
        const double ref_kappa = frames.front().pos.kappa;

        double min_x = std::numeric_limits<double>::max();
        double min_y = std::numeric_limits<double>::max();
        double max_x = std::numeric_limits<double>::lowest();
        double max_y = std::numeric_limits<double>::lowest();

        for (size_t i = 0; i < frames.size(); ++i) {
            cv::Mat img = cv::imread(frames[i].path, cv::IMREAD_COLOR);
            if (img.empty()) {
                output.message = "failed to read image: " + frames[i].path;
                return output;
            }
            if (compose_scale < 0.999) {
                cv::resize(img, img, cv::Size(), compose_scale, compose_scale, cv::INTER_AREA);
            }
            const double yaw_delta = normalizeAngleDeg(frames[i].pos.kappa - ref_kappa);
            img = rotateImageExpand(img, -yaw_delta);
            images.push_back(img);
            cv::Mat gray;
            cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
            gray.convertTo(gray, CV_32F);
            grays.push_back(gray);

            pos_centers[i] = {local_coords[i].x * pixels_per_meter, -local_coords[i].y * pixels_per_meter};
        }

        const std::vector<std::pair<size_t, size_t> > strips = buildStripRanges(pos_centers);
        for (const auto &[s, e]: strips) {
            if (s >= e) {
                continue;
            }
            centers[s] = pos_centers[s];
            for (size_t i = s + 1; i < e; ++i) {
                const cv::Point2d predicted = pos_centers[i] - pos_centers[i - 1];
                double response = 0.0;
                cv::Point2d shift;
                cv::Point2d measured = predicted;
                if (phaseCorrelateSafe(grays[i - 1], grays[i], shift, response)) {
                    measured = {-shift.x, -shift.y};
                } else {
                    response = 0.0;
                }

                // 视觉平移与POS先验差异过大时，抑制视觉估计，防止错配导致漂移。
                const double pred_norm = cv::norm(predicted);
                const double diff = cv::norm(measured - predicted);
                bool visual_ok = response >= 0.35;
                if (visual_ok) {
                    const double gate = std::max(8.0, pred_norm * 0.85 + 8.0);
                    visual_ok = diff <= gate;
                }
                if (!visual_ok) {
                    measured = predicted;
                }

                const cv::Point2d fused_delta = predicted * 0.90 + measured * 0.10;
                centers[i] = centers[i - 1] + fused_delta;
            }

            // 每个航带独立线性回拉到 POS 锚点，避免跨带累计漂移。
            if (e - s == 1) {
                centers[s] = pos_centers[s];
            } else {
                const cv::Point2d err_start = pos_centers[s] - centers[s];
                const cv::Point2d err_end = pos_centers[e - 1] - centers[e - 1];
                const double denom = static_cast<double>((e - s) - 1);
                for (size_t i = s; i < e; ++i) {
                    const double t = static_cast<double>(i - s) / denom;
                    centers[i] += err_start * (1.0 - t) + err_end * t;
                }
            }
        }

        for (size_t i = 0; i < images.size(); ++i) {
            const auto &img = images[i];
            const double tl_x = centers[i].x - img.cols * 0.5;
            const double tl_y = centers[i].y - img.rows * 0.5;
            min_x = std::min(min_x, tl_x);
            min_y = std::min(min_y, tl_y);
            max_x = std::max(max_x, tl_x + img.cols);
            max_y = std::max(max_y, tl_y + img.rows);
        }

        const int canvas_w = static_cast<int>(std::ceil(max_x - min_x));
        const int canvas_h = static_cast<int>(std::ceil(max_y - min_y));
        if (canvas_w <= 0 || canvas_h <= 0) {
            output.message = "invalid output canvas for POS-projection mode";
            return output;
        }

        cv::Mat best_img(canvas_h, canvas_w, CV_8UC3, cv::Scalar::all(0));
        cv::Mat best_w(canvas_h, canvas_w, CV_32F, cv::Scalar::all(0));

        std::cout << "[STITCH] fallback POS projection mode: " << reason
                << ", compose_scale=" << compose_scale
                << ", ppm=" << pixels_per_meter
                << ", canvas=" << canvas_w << "x" << canvas_h << std::endl;

        cv::Size cached_size;
        cv::Mat cached_weight;
        for (size_t i = 0; i < images.size(); ++i) {
            const auto &img = images[i];
            if (cached_size != img.size()) {
                cached_weight = buildFeatherWeight(img.size());
                cached_size = img.size();
            }

            const int x = static_cast<int>(std::lround(centers[i].x - img.cols * 0.5 - min_x));
            const int y = static_cast<int>(std::lround(centers[i].y - img.rows * 0.5 - min_y));
            const cv::Rect roi(x, y, img.cols, img.rows);
            if (roi.x < 0 || roi.y < 0 || roi.br().x > canvas_w || roi.br().y > canvas_h) {
                continue;
            }

            cv::Mat roi_best_w = best_w(roi);
            cv::Mat better_mask;
            cv::compare(cached_weight, roi_best_w, better_mask, cv::CMP_GT);

            cached_weight.copyTo(roi_best_w, better_mask);
            img.copyTo(best_img(roi), better_mask);
        }

        fs::path out_path(output_path);
        fs::create_directories(out_path.parent_path());
        if (!cv::imwrite(output_path, best_img)) {
            output.message = "failed to write output image: " + output_path;
            return output;
        }

        output.ok = true;
        output.message = "stitch completed (POS projection fallback)";
        return output;
    }
}

StitchOutput stitchWithPosGuidance(const std::vector<StitchFrame> &frames,
                                   const StitchTuning &tuning,
                                   const std::string &output_path) {
    cv::ocl::setUseOpenCL(false);

    StitchOutput output;
    output.input_count = static_cast<int>(frames.size());
    output.output_path = output_path;
    if (frames.size() < 2) {
        output.message = "need at least 2 frames after POS alignment";
        return output;
    }
    if (frames.size() > 180) {
        StitchOutput strip_result = stitchByStripMosaics(frames, tuning, output_path, "large batch optimization");
        if (strip_result.ok) {
            return strip_result;
        }
        return stitchByPosProjection(frames, tuning, output_path, "large batch optimization");
    }

    try {
        const double lon0 = frames.front().pos.longitude;
        const double lat0 = frames.front().pos.latitude;

        std::vector<cv::Point2d> local_coords;
        local_coords.reserve(frames.size());
        for (const auto &frame: frames) {
            local_coords.push_back(lonLatToLocal(frame.pos, lon0, lat0));
        }

        std::vector<cv::detail::ImageFeatures> features(frames.size());
        std::vector<cv::Size> full_img_sizes(frames.size());

        cv::Ptr<cv::Feature2D> finder = cv::SIFT::create(std::max(500, tuning.sift_features));
        double work_scale = 1.0;
        bool work_scale_set = false;

        std::cout << "[STITCH] computing features for " << frames.size() << " images..." << std::endl;
        for (size_t i = 0; i < frames.size(); ++i) {
            cv::Mat full_img = cv::imread(frames[i].path, cv::IMREAD_COLOR);
            if (full_img.empty()) {
                throw std::runtime_error("failed to read image: " + frames[i].path);
            }
            full_img_sizes[i] = full_img.size();
            if (!work_scale_set) {
                work_scale = calcImageScaleMpx(full_img.size(), tuning.registration_resol_mpx);
                work_scale_set = true;
            }

            cv::Mat img_for_features;
            if (work_scale < 0.999) {
                cv::resize(full_img, img_for_features, cv::Size(), work_scale, work_scale, cv::INTER_LINEAR);
            } else {
                img_for_features = full_img;
            }

            cv::detail::computeImageFeatures(finder, img_for_features, features[i]);
            features[i].img_idx = static_cast<int>(i);
            std::cout << "  [feat] " << (i + 1) << "/" << frames.size()
                    << " id=" << frames[i].id
                    << " kpts=" << features[i].keypoints.size() << std::endl;
        }

        cv::UMat matching_mask = buildMatchingMask(local_coords, tuning);

        std::vector<cv::detail::MatchesInfo> pairwise_matches;
        cv::Ptr<cv::detail::FeaturesMatcher> matcher;
        if (tuning.use_affine_bundle) {
            matcher = cv::makePtr<cv::detail::AffineBestOf2NearestMatcher>(
                    false, tuning.try_gpu, tuning.match_conf, tuning.min_good_matches);
        } else {
            matcher = cv::makePtr<cv::detail::BestOf2NearestMatcher>(
                    tuning.try_gpu, tuning.match_conf, tuning.min_good_matches, tuning.min_good_matches);
        }
        std::cout << "[STITCH] matching candidate pairs..." << std::endl;
        (*matcher)(features, pairwise_matches, matching_mask);
        matcher->collectGarbage();

        std::vector<int> component = cv::detail::leaveBiggestComponent(
                features, pairwise_matches, tuning.pano_conf_thresh);
        if (component.size() < 2) {
            StitchOutput strip_result = stitchByStripMosaics(frames, tuning, output_path, "no connected component");
            if (strip_result.ok) {
                return strip_result;
            }
            return stitchByPosProjection(frames, tuning, output_path, "no connected component");
        }
        if (component.size() < static_cast<size_t>(std::max(2, static_cast<int>(frames.size() * 0.80)))) {
            StitchOutput strip_result = stitchByStripMosaics(
                    frames, tuning, output_path,
                    "largest component too small: " + std::to_string(component.size()) + "/" +
                    std::to_string(frames.size()));
            if (strip_result.ok) {
                return strip_result;
            }
            return stitchByPosProjection(
                    frames, tuning, output_path,
                    "largest component too small: " + std::to_string(component.size()) + "/" +
                    std::to_string(frames.size()));
        }

        std::vector<StitchFrame> used_frames = subsetByIndices(frames, component);
        std::vector<cv::Size> used_sizes = subsetByIndices(full_img_sizes, component);
        output.stitched_count = static_cast<int>(used_frames.size());
        std::cout << "[STITCH] connected component size=" << used_frames.size() << std::endl;

        cv::Ptr<cv::detail::Estimator> estimator;
        if (tuning.use_affine_bundle) {
            estimator = cv::makePtr<cv::detail::AffineBasedEstimator>();
        } else {
            estimator = cv::makePtr<cv::detail::HomographyBasedEstimator>();
        }

        std::vector<cv::detail::CameraParams> cameras;
        if (!(*estimator)(features, pairwise_matches, cameras)) {
            output.message = "camera parameter estimation failed";
            return output;
        }
        for (auto &camera: cameras) {
            if (camera.R.empty()) {
                camera.R = cv::Mat::eye(3, 3, CV_32F);
            } else {
                camera.R.convertTo(camera.R, CV_32F);
            }
        }

        cv::Ptr<cv::detail::BundleAdjusterBase> adjuster;
        if (tuning.use_affine_bundle) {
            adjuster = cv::makePtr<cv::detail::BundleAdjusterAffinePartial>();
        } else {
            adjuster = cv::makePtr<cv::detail::BundleAdjusterRay>();
        }
        adjuster->setConfThresh(tuning.pano_conf_thresh);
        cv::Mat_<uchar> refine_mask = cv::Mat::ones(3, 3, CV_8U);
        adjuster->setRefinementMask(refine_mask);
        if (!(*adjuster)(features, pairwise_matches, cameras)) {
            output.message = "bundle adjustment failed";
            return output;
        }

        for (auto &camera: cameras) {
            camera.R.convertTo(camera.R, CV_32F);
        }

        std::vector<double> focals;
        focals.reserve(cameras.size());
        for (const auto &camera: cameras) {
            focals.push_back(camera.focal);
        }
        std::ranges::sort(focals);
        float warped_image_scale = 1.0f;
        if (!focals.empty() && !tuning.use_affine_warper) {
            const size_t mid = focals.size() / 2;
            warped_image_scale = static_cast<float>(focals[mid]);
        }

        double seam_scale = calcImageScaleMpx(used_sizes.front(), tuning.seam_estimation_resol_mpx);
        double compose_scale = tuning.compositing_resol_mpx <= 0
                                   ? 1.0
                                   : calcImageScaleMpx(used_sizes.front(), tuning.compositing_resol_mpx);

        // 对 400 张级别默认降一点输出分辨率，避免内存爆炸。
        if (compose_scale > 0.999 && used_frames.size() > 180) {
            compose_scale = std::max(0.35, seam_scale);
            std::cout << "[STITCH] large batch detected, auto compose_scale=" << compose_scale << std::endl;
        }

        const float seam_aspect = static_cast<float>(seam_scale / work_scale);
        cv::Ptr<cv::detail::RotationWarper> seam_warper = makeWarper(
                tuning, warped_image_scale * seam_aspect);

        std::vector<cv::Point> corners(used_frames.size());
        std::vector<cv::Size> sizes(used_frames.size());
        std::vector<cv::UMat> images_warped(used_frames.size());
        std::vector<cv::UMat> masks_warped(used_frames.size());
        std::vector<cv::UMat> seam_images_f(used_frames.size());

        std::cout << "[STITCH] seam preparation..." << std::endl;
        for (size_t i = 0; i < used_frames.size(); ++i) {
            cv::Mat full_img = cv::imread(used_frames[i].path, cv::IMREAD_COLOR);
            if (full_img.empty()) {
                throw std::runtime_error("failed to read image: " + used_frames[i].path);
            }

            cv::Mat seam_img;
            if (seam_scale < 0.999) {
                cv::resize(full_img, seam_img, cv::Size(), seam_scale, seam_scale, cv::INTER_LINEAR);
            } else {
                seam_img = full_img;
            }

            cv::Mat mask(seam_img.size(), CV_8U, cv::Scalar::all(255));
            cv::Mat K;
            cameras[i].K().convertTo(K, CV_32F);
            K.at<float>(0, 0) *= seam_aspect;
            K.at<float>(0, 2) *= seam_aspect;
            K.at<float>(1, 1) *= seam_aspect;
            K.at<float>(1, 2) *= seam_aspect;

            corners[i] = seam_warper->warp(seam_img, K, cameras[i].R, cv::INTER_LINEAR, cv::BORDER_REFLECT,
                                           images_warped[i]);
            seam_warper->warp(mask, K, cameras[i].R, cv::INTER_NEAREST, cv::BORDER_CONSTANT, masks_warped[i]);
            sizes[i] = images_warped[i].size();
            images_warped[i].convertTo(seam_images_f[i], CV_32F);
        }

        cv::Ptr<cv::detail::ExposureCompensator> compensator;
        if (tuning.use_blocks_gain) {
            compensator = cv::makePtr<cv::detail::BlocksGainCompensator>(32, 32);
        } else {
            compensator = cv::makePtr<cv::detail::GainCompensator>();
        }
        compensator->feed(corners, images_warped, masks_warped);

        cv::Ptr<cv::detail::SeamFinder> seam_finder = cv::makePtr<cv::detail::VoronoiSeamFinder>();
        seam_finder->find(seam_images_f, corners, masks_warped);

        const float compose_aspect = static_cast<float>(compose_scale / work_scale);
        cv::Ptr<cv::detail::RotationWarper> compose_warper = makeWarper(
                tuning, warped_image_scale * compose_aspect);

        std::vector<cv::Point> compose_corners(used_frames.size());
        std::vector<cv::Size> compose_sizes(used_frames.size());
        for (size_t i = 0; i < used_frames.size(); ++i) {
            cv::Mat K;
            cameras[i].K().convertTo(K, CV_32F);
            K.at<float>(0, 0) *= compose_aspect;
            K.at<float>(0, 2) *= compose_aspect;
            K.at<float>(1, 1) *= compose_aspect;
            K.at<float>(1, 2) *= compose_aspect;
            compose_corners[i] = compose_warper->warpRoi(used_sizes[i], K, cameras[i].R).tl();
            compose_sizes[i] = compose_warper->warpRoi(used_sizes[i], K, cameras[i].R).size();
        }

        cv::Ptr<cv::detail::Blender> blender = cv::detail::Blender::createDefault(
                cv::detail::Blender::MULTI_BAND, tuning.try_gpu);
        if (auto *mb = dynamic_cast<cv::detail::MultiBandBlender *>(blender.get())) {
            mb->setNumBands(std::max(1, tuning.blend_bands));
        }
        blender->prepare(cv::detail::resultRoi(compose_corners, compose_sizes));

        std::cout << "[STITCH] blending and composing..." << std::endl;
        for (size_t i = 0; i < used_frames.size(); ++i) {
            cv::Mat full_img = cv::imread(used_frames[i].path, cv::IMREAD_COLOR);
            if (full_img.empty()) {
                throw std::runtime_error("failed to read image: " + used_frames[i].path);
            }

            if (compose_scale < 0.999) {
                cv::resize(full_img, full_img, cv::Size(), compose_scale, compose_scale, cv::INTER_LINEAR);
            }

            cv::Mat K;
            cameras[i].K().convertTo(K, CV_32F);
            K.at<float>(0, 0) *= compose_aspect;
            K.at<float>(0, 2) *= compose_aspect;
            K.at<float>(1, 1) *= compose_aspect;
            K.at<float>(1, 2) *= compose_aspect;

            cv::Mat img_warped;
            cv::Mat mask(full_img.size(), CV_8U, cv::Scalar::all(255));
            cv::Mat mask_warped;
            const cv::Point corner = compose_warper->warp(
                    full_img, K, cameras[i].R, cv::INTER_LINEAR, cv::BORDER_REFLECT, img_warped);
            compose_warper->warp(mask, K, cameras[i].R, cv::INTER_NEAREST, cv::BORDER_CONSTANT, mask_warped);

            compensator->apply(static_cast<int>(i), corner, img_warped, mask_warped);

            cv::Mat seam_mask;
            masks_warped[i].copyTo(seam_mask);
            if (seam_mask.size() != mask_warped.size()) {
                cv::resize(seam_mask, seam_mask, mask_warped.size(), 0, 0, cv::INTER_LINEAR_EXACT);
            }
            cv::bitwise_and(mask_warped, seam_mask, mask_warped);

            cv::Mat img_warped_s16;
            img_warped.convertTo(img_warped_s16, CV_16S);
            blender->feed(img_warped_s16, mask_warped, corner);
        }

        cv::Mat result;
        cv::Mat result_mask;
        blender->blend(result, result_mask);

        fs::path out_path(output_path);
        fs::create_directories(out_path.parent_path());
        if (!cv::imwrite(output_path, result)) {
            output.message = "failed to write output image: " + output_path;
            return output;
        }

        output.ok = true;
        output.message = "stitch completed";
        return output;
    } catch (const std::exception &e) {
        StitchOutput strip_result = stitchByStripMosaics(
                frames, tuning, output_path, std::string("detail pipeline exception: ") + e.what());
        if (strip_result.ok) {
            return strip_result;
        }
        return stitchByPosProjection(frames, tuning, output_path, std::string("detail pipeline exception: ") + e.what());
    }
}
