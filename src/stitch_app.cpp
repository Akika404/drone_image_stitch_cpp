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
#include <utility>
#include <vector>

#include "image_loader.hpp"
#include "stitch_common.hpp"
#include "stitch_config.hpp"
#include "stitch_global.hpp"
#include "stitch_robust.hpp"
#include "visual_flight_grouper.hpp"

namespace fs = std::filesystem;

namespace {
    const CameraCalibration *findCameraCalibration(
        const StitchTuning &tuning,
        const std::string &camera_id) {
        for (const auto &cam: tuning.calibration.cameras) {
            if (cam.camera_id == camera_id) {
                return &cam;
            }
        }
        return nullptr;
    }

    bool undistortImagesIfReady(
        std::vector<cv::Mat> &images,
        const CameraCalibration &cam,
        const std::string &image_type) {
        if (!cam.isMetricReady()) {
            std::cout << "[Main] undistort skipped for '" << image_type
                    << "': calibration incomplete" << std::endl;
            return false;
        }
        if (images.empty()) {
            return false;
        }

        if (cam.image_width > 0 && cam.image_height > 0) {
            const cv::Size expected(cam.image_width, cam.image_height);
            if (images.front().size() != expected) {
                std::cout << "[Main] undistort skipped for '" << image_type
                        << "': image size mismatch, expected " << expected.width << "x" << expected.height
                        << ", got " << images.front().cols << "x" << images.front().rows << std::endl;
                return false;
            }
        }

        cv::Mat K = (cv::Mat_<double>(3, 3) <<
            *cam.fx_px, 0.0, *cam.cx_px,
            0.0, *cam.fy_px, *cam.cy_px,
            0.0, 0.0, 1.0);

        cv::Mat dist(1, 8, CV_64F);
        for (int i = 0; i < 8; ++i) {
            dist.at<double>(0, i) = cam.distortion->at(i);
        }

        for (auto &img: images) {
            cv::Mat undistorted;
            cv::undistort(img, undistorted, K, dist);
            img = std::move(undistorted);
        }

        std::cout << "[Main] undistort applied for '" << image_type
                << "' (" << images.size() << " images)" << std::endl;
        return true;
    }

    void logRuntimeOptions(const StitchTuning &tuning) {
        const int strip_sift = tuning.strip_sift_features > 0 ? tuning.strip_sift_features : tuning.sift_features;
        const int global_sift = tuning.global_sift_features > 0 ? tuning.global_sift_features : tuning.sift_features;
        std::cout << "[Main] opencl: requested=" << (tuning.use_opencl ? "on" : "off")
                << ", enabled=" << (cv::ocl::useOpenCL() ? "yes" : "no")
                << ", try_gpu=" << (tuning.try_gpu ? "on" : "off") << std::endl;
        if (cv::ocl::useOpenCL()) {
            const cv::ocl::Device &dev = cv::ocl::Device::getDefault();
            std::cout << "[Main] opencl device: " << dev.vendorName() << " / " << dev.name() << std::endl;
        }
        std::cout << "[Main] stitch params: sift=" << tuning.sift_features
                << ", strip_sift=" << strip_sift
                << ", global_sift=" << global_sift
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
        std::cout << "[Main] calibration metric-ready: "
                << (tuning.calibration.anyMetricReady() ? "yes" : "no") << std::endl;
        for (const auto &cam: tuning.calibration.cameras) {
            std::cout << "[Main]   cam=" << cam.camera_id
                    << ", intrinsics=" << (cam.hasIntrinsics() ? "yes" : "no")
                    << ", distortion=" << (cam.hasDistortion() ? "yes" : "no")
                    << std::endl;
        }
    }

    void flattenStripGroups(
        const std::vector<VisualStripGroup> &strip_groups,
        std::vector<cv::Mat> &all_images,
        std::vector<std::string> &all_tags) {
        for (size_t i = 0; i < strip_groups.size(); ++i) {
            for (size_t j = 0; j < strip_groups[i].images.size(); ++j) {
                all_images.push_back(strip_groups[i].images[j]);
                if (j < strip_groups[i].image_ids.size() && !strip_groups[i].image_ids[j].empty()) {
                    all_tags.push_back(strip_groups[i].image_ids[j]);
                } else {
                    all_tags.push_back("img#" + std::to_string(all_images.size() - 1));
                }
            }
        }
    }

    std::vector<std::string> makeStripTags(const VisualStripGroup &group) {
        std::vector<std::string> tags;
        tags.reserve(group.images.size());
        for (size_t i = 0; i < group.images.size(); ++i) {
            if (i < group.image_ids.size() && !group.image_ids[i].empty()) {
                tags.push_back(group.image_ids[i]);
            } else {
                tags.push_back("img#" + std::to_string(i));
            }
        }
        return tags;
    }

} // namespace

int runStitchApplication() {
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);

    const std::string image_folder = "../images";
    const std::string image_type = "visible";
    const std::string group = "minfull";
    const StitchTuning tuning = loadStitchTuning(image_type);

    const std::string input_folder = image_folder + "/" + image_type + "/" + group;
    const std::string output_folder = "../output/" + image_type + "/" + group;
    fs::create_directories(output_folder);

    const std::string filename = image_type + "_" + group + "_" + "uav_panorama.jpg";
    const fs::path output_path = fs::path(output_folder) / filename;
    fs::create_directories(output_path.parent_path() / "strips");

    try {
        std::cout << "[Main] input dir: " << input_folder << std::endl;
        std::cout << "[Main] stitch mode: visual-only (no POS / no EXIF geo)" << std::endl;
        std::cout << "[Main] output: " << output_path << std::endl;
        logRuntimeOptions(tuning);

        auto [images, ids] = ImageLoader::loadWithIds(input_folder);
        std::cout << "[Main] valid images: " << images.size() << std::endl;
        if (images.size() < 2) {
            throw std::runtime_error("need at least 2 images to stitch");
        }

        if (const auto *cam = findCameraCalibration(tuning, image_type)) {
            undistortImagesIfReady(images, *cam, image_type);
        } else {
            std::cout << "[Main] undistort skipped for '" << image_type
                    << "': no camera_id entry in tuning.calibration.cameras" << std::endl;
        }

        auto strip_groups = VisualFlightGrouper::groupBoustrophedon(images, ids, tuning);
        if (strip_groups.empty()) {
            throw std::runtime_error("visual grouping produced no valid strips");
        }

        cv::Mat panorama;
        if (strip_groups.size() > 1) {
            std::cout << "[Main] multi-strip mode, preserving visual flight order..." << std::endl;
            for (size_t i = 0; i < strip_groups.size(); ++i) {
                std::cout << "[Main]   strip " << i << ": " << strip_groups[i].images.size() << " images" << std::endl;
            }

            std::vector<cv::Mat> strip_panoramas;
            strip_panoramas.reserve(strip_groups.size());
            fs::path strips_dir = output_path.parent_path() / "strips";
            fs::create_directories(strips_dir);

            StitchTuning strip_tuning = tuning;
            strip_tuning.sift_features = tuning.strip_sift_features > 0
                ? tuning.strip_sift_features
                : tuning.sift_features;
            for (size_t si = 0; si < strip_groups.size(); ++si) {
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
            global_tuning.sift_features = tuning.global_sift_features > 0
                ? tuning.global_sift_features
                : tuning.sift_features;
            global_tuning.use_range_matcher = false;
            global_tuning.range_width = 2;

            // 可以取消注释以加快速度，但是图像清晰度会明显下降
            // if (global_tuning.compositing_resol_mpx < 0.0) {
            //     global_tuning.compositing_resol_mpx = 2.0;
            // }

            global_tuning.blend_bands = std::max(global_tuning.blend_bands, 5);
            std::cout << "[Main] global-stage: strip_panoramas=" << strip_panoramas.size()
                    << ", sift=" << global_tuning.sift_features
                    << ", compose_mpx=" << global_tuning.compositing_resol_mpx
                    << ", blend_bands=" << global_tuning.blend_bands << std::endl;

            panorama = stitchInterStripsCustom(strip_panoramas, global_tuning);
        } else {
            std::vector<cv::Mat> all_images;
            std::vector<std::string> all_tags;
            flattenStripGroups(strip_groups, all_images, all_tags);
            if (all_images.size() < 2) {
                throw std::runtime_error("need at least 2 images to stitch");
            }
            std::cout << "[Main] single-group stitch: " << all_images.size() << " images" << std::endl;
            StitchTuning single_group_tuning = tuning;
            single_group_tuning.sift_features = tuning.strip_sift_features > 0
                ? tuning.strip_sift_features
                : tuning.sift_features;
            panorama = stitchRobustly(
                all_images, cv::Stitcher::SCANS, "Stitch", single_group_tuning, single_group_tuning.range_width, &all_tags);
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
