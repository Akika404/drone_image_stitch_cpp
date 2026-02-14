#include "stitch_app.hpp"

#include <opencv2/core/ocl.hpp>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "image_loader.hpp"
#include "pos_image_grouper.hpp"
#include "pos_reader.hpp"
#include "stitch_common.hpp"
#include "stitch_config.hpp"
#include "stitch_global.hpp"
#include "stitch_robust.hpp"

namespace fs = std::filesystem;

namespace {
    void logRuntimeOptions(const StitchTuning &tuning) {
        std::cout << "[Main] opencl: requested=" << (tuning.use_opencl ? "on" : "off")
                << ", enabled=" << (cv::ocl::useOpenCL() ? "yes" : "no")
                << ", try_gpu=" << (tuning.try_gpu ? "on" : "off") << std::endl;
        if (cv::ocl::useOpenCL()) {
            const cv::ocl::Device &dev = cv::ocl::Device::getDefault();
            std::cout << "[Main] opencl device: " << dev.vendorName() << " / " << dev.name() << std::endl;
        }
        std::cout << "[Main] stitch params: sift=" << tuning.sift_features
                << ", match_conf=" << tuning.match_conf
                << ", range_matcher=" << (tuning.use_range_matcher ? "on" : "off")
                << ", range_width=" << tuning.range_width
                << ", affine_bundle=" << (tuning.use_affine_bundle ? "on" : "off")
                << ", affine_warper=" << (tuning.use_affine_warper ? "on" : "off")
                << ", anchor_fallback=" << (tuning.use_anchor_fallback ? "on" : "off")
                << ", anchor_window=" << tuning.anchor_window
                << ", reg_mpx=" << tuning.registration_resol_mpx
                << ", seam_mpx=" << tuning.seam_estimation_resol_mpx
                << ", compose_mpx=" << tuning.compositing_resol_mpx << std::endl;
    }

    void flattenStripGroups(
        const std::vector<FlightStripGroup> &strip_groups,
        std::vector<cv::Mat> &all_images,
        std::vector<std::string> &all_tags) {
        for (size_t i = 0; i < strip_groups.size(); ++i) {
            for (size_t j = 0; j < strip_groups[i].images.size(); ++j) {
                all_images.push_back(strip_groups[i].images[j]);
                if (j < strip_groups[i].records.size() && !strip_groups[i].records[j].file_id.empty()) {
                    all_tags.push_back(strip_groups[i].records[j].file_id);
                } else {
                    all_tags.push_back("img#" + std::to_string(all_images.size() - 1));
                }
            }
        }
    }

    std::vector<std::string> makeStripTags(const FlightStripGroup &group) {
        std::vector<std::string> tags;
        tags.reserve(group.images.size());
        for (size_t i = 0; i < group.images.size(); ++i) {
            if (i < group.records.size() && !group.records[i].file_id.empty()) {
                tags.push_back(group.records[i].file_id);
            } else {
                tags.push_back("img#" + std::to_string(i));
            }
        }
        return tags;
    }

    cv::UMat makeAdjacentStripMask(const int n, const int neighbor_width) {
        cv::Mat mask(n, n, CV_8U, cv::Scalar(0));
        for (int i = 0; i < n; ++i) {
            for (int j = std::max(0, i - neighbor_width); j <= std::min(n - 1, i + neighbor_width); ++j) {
                if (i != j) {
                    mask.at<uchar>(i, j) = 1;
                }
            }
        }
        cv::UMat umask;
        mask.copyTo(umask);
        return umask;
    }

    void removeRedundantImages(FlightStripGroup &group) {
        if (group.records.size() != group.images.size() || group.records.size() < 2) {
            return;
        }

        const double R = 6378137.0; // Earth radius in meters
        const double to_rad = CV_PI / 180.0;
        
        std::vector<double> dists;
        dists.reserve(group.records.size() - 1);

        for (size_t i = 0; i < group.records.size() - 1; ++i) {
            const auto &p1 = group.records[i];
            const auto &p2 = group.records[i+1];
            
            double dlat = (p2.latitude - p1.latitude) * to_rad;
            double dlon = (p2.longitude - p1.longitude) * to_rad;
            double lat_avg = (p1.latitude + p2.latitude) / 2.0 * to_rad;
            
            double x = dlon * std::cos(lat_avg);
            double y = dlat;
            double d = std::sqrt(x*x + y*y) * R;
            dists.push_back(d);
        }

        std::vector<double> sorted_dists = dists;
        std::sort(sorted_dists.begin(), sorted_dists.end());
        double median_dist = 0;
        if (!sorted_dists.empty()) {
            median_dist = sorted_dists[sorted_dists.size() / 2];
        }

        if (median_dist < 1.0) {
             return;
        }

        double threshold = median_dist * 0.3; 
        
        std::vector<cv::Mat> new_images;
        std::vector<PosRecord> new_records;
        new_images.reserve(group.images.size());
        new_records.reserve(group.records.size());
        
        new_images.push_back(group.images[0]);
        new_records.push_back(group.records[0]);

        int removed_count = 0;
        for (size_t i = 1; i < group.records.size(); ++i) {
             const auto &p1 = new_records.back(); 
             const auto &p2 = group.records[i];

             double dlat = (p2.latitude - p1.latitude) * to_rad;
             double dlon = (p2.longitude - p1.longitude) * to_rad;
             double lat_avg = (p1.latitude + p2.latitude) / 2.0 * to_rad;
             double x = dlon * std::cos(lat_avg);
             double y = dlat;
             double d = std::sqrt(x*x + y*y) * R;

             if (d >= threshold) {
                 new_images.push_back(group.images[i]);
                 new_records.push_back(group.records[i]);
             } else {
                 removed_count++;
             }
        }

        if (removed_count > 0) {
            std::cout << "[Main] Filtered " << removed_count << " redundant images (hovering/overlapping) from strip." << std::endl;
            group.images = new_images;
            group.records = new_records;
        }
    }

} // namespace

int runStitchApplication() {
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);
    const StitchTuning tuning = loadStitchTuning();

    constexpr std::string image_folder = "../images";
    constexpr std::string image_type = "visible";
    constexpr std::string group = "full";
    constexpr std::string pos_path = "../assets/pos.mti";

    constexpr bool use_pos = true;

    constexpr std::string input_folder = image_folder + "/" + image_type + "/" + group;
    const std::string output_folder = "../output/" + image_type + "/" + group;
    fs::create_directories(output_folder);

    const std::string filename = image_type + "_" + group + "_" + "uav_panorama.jpg";
    const fs::path output_path = fs::path(output_folder) / filename;
    fs::create_directories(output_path.parent_path() / "strips");

    try {
        std::cout << "[Main] input dir: " << input_folder << std::endl;
        std::cout << "[Main] POS: " << (use_pos ? "enabled" : "disabled") << std::endl;
        std::cout << "[Main] output: " << output_path << std::endl;
        logRuntimeOptions(tuning);

        const auto [images, ids] = ImageLoader::loadWithIds(input_folder);
        std::cout << "[Main] valid images: " << images.size() << std::endl;
        if (images.size() < 2) {
            throw std::runtime_error("need at least 2 images to stitch");
        }

        std::vector<FlightStripGroup> strip_groups;
        if (use_pos) {
            std::cout << "[Main] POS file: " << pos_path << std::endl;
            const auto pos_records = PosReader::load(pos_path);
            strip_groups = PosBasedImageGrouper::groupWithRecords(images, ids, pos_records);
            if (strip_groups.empty()) {
                throw std::runtime_error("POS grouping empty, check POS file and image IDs");
            }
        } else {
            strip_groups.push_back({images, {}});
            std::cout << "[Main] POS disabled, all images as single group" << std::endl;
        }

        cv::Mat panorama;
        if (use_pos && strip_groups.size() > 1) {
            std::cout << "[Main] multi-strip mode, ordering by cross-track position..." << std::endl;
            orderStripGroupsByCrossTrack(strip_groups);
            for (size_t i = 0; i < strip_groups.size(); ++i) {
                std::cout << "[Main]   strip " << i << ": " << strip_groups[i].images.size() << " images" << std::endl;
            }

            std::vector<cv::Mat> strip_panoramas;
            strip_panoramas.reserve(strip_groups.size());
            fs::path strips_dir = output_path.parent_path() / "strips";
            fs::create_directories(strips_dir);

            StitchTuning strip_tuning = tuning;
            for (size_t si = 0; si < strip_groups.size(); ++si) {
                // 偏航照片过滤
                removeRedundantImages(strip_groups[si]);
                std::cout << "[Main] strip-stage: stitching strip " << si
                        << " (" << strip_groups[si].images.size() << " images)..." << std::endl;
                auto strip_tags = makeStripTags(strip_groups[si]);
                cv::Mat strip_pano = stitchRobustly(
                    strip_groups[si].images,
                    cv::Stitcher::SCANS,
                    "Strip" + std::to_string(si),
                    strip_tuning,
                    strip_tuning.range_width,
                    &strip_tags);
                autoCropBlackBorder(strip_pano);

                std::ostringstream strip_name;
                strip_name << "strip_" << std::setw(2) << std::setfill('0') << si << ".jpg";
                cv::imwrite((strips_dir / strip_name.str()).string(), strip_pano);
                std::cout << "[Main] strip-stage: strip " << si << " panorama="
                        << strip_pano.cols << "x" << strip_pano.rows << std::endl;
                strip_panoramas.push_back(std::move(strip_pano));
            }

            if (strip_panoramas.size() < 2) {
                throw std::runtime_error("need at least 2 strip panoramas for multi-strip compose");
            }

            StitchTuning global_tuning = tuning;
            global_tuning.use_range_matcher = false;
            global_tuning.range_width = 2;

            // 可以取消注释以加快速度，但是图像清晰度会明显下降
            // if (global_tuning.compositing_resol_mpx < 0.0) {
            //     global_tuning.compositing_resol_mpx = 2.0;
            // }

            global_tuning.blend_bands = std::min(global_tuning.blend_bands, 3);
            std::cout << "[Main] global-stage: strip_panoramas=" << strip_panoramas.size()
                    << ", compose_mpx=" << global_tuning.compositing_resol_mpx
                    << ", blend_bands=" << global_tuning.blend_bands << std::endl;

            cv::UMat strip_mask = makeAdjacentStripMask(static_cast<int>(strip_panoramas.size()), 1);
            panorama = stitchRobustly(
                strip_panoramas,
                cv::Stitcher::SCANS,
                "Global",
                global_tuning,
                global_tuning.range_width,
                nullptr,
                &strip_mask);
        } else {
            std::vector<cv::Mat> all_images;
            std::vector<std::string> all_tags;
            flattenStripGroups(strip_groups, all_images, all_tags);
            if (all_images.size() < 2) {
                throw std::runtime_error("need at least 2 images to stitch");
            }
            std::cout << "[Main] single-group stitch: " << all_images.size() << " images" << std::endl;
            panorama = stitchRobustly(
                all_images, cv::Stitcher::SCANS, "Stitch", tuning, tuning.range_width, &all_tags);
        }

        autoCropBlackBorder(panorama);
        cv::imwrite(output_path.string(), panorama);
        std::cout << "[Finish] done: " << output_path << std::endl;
    } catch (const std::exception &e) {
        std::cerr << "[Error] " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
