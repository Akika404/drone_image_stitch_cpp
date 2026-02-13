#include "stitch_app.hpp"

#include <opencv2/core/ocl.hpp>
#include <opencv2/opencv.hpp>

#include <filesystem>
#include <iostream>
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
} // namespace

int runStitchApplication() {
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);
    const StitchTuning tuning = loadStitchTuning();

    const std::string image_folder = "../images";
    const std::string image_type = "visible";
    const std::string group = "full";
    const std::string pos_path = "../assets/pos.mti";

    const bool use_pos = true;

    const std::string input_folder = image_folder + "/" + image_type + "/" + group;
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

            const auto input = buildGlobalStitchInput(strip_groups, tuning);
            std::cout << "[Main] global pipeline: total_images=" << input.images.size()
                    << ", within_strip_pairs=" << input.num_within_pairs
                    << ", cross_strip_pairs=" << input.num_cross_pairs << std::endl;

            StitchTuning global_tuning = tuning;
            // Multi-band blending at full compose resolution is expensive for multi-strip panoramas.
            if (global_tuning.compositing_resol_mpx < 0.0) {
                global_tuning.compositing_resol_mpx = 2.0;
            }
            global_tuning.blend_bands = std::min(global_tuning.blend_bands, 3);
            std::cout << "[Main] global speed tuning: compose_mpx=" << global_tuning.compositing_resol_mpx
                    << ", blend_bands=" << global_tuning.blend_bands << std::endl;

            panorama = stitchRobustly(
                input.images,
                cv::Stitcher::SCANS,
                "Global",
                global_tuning,
                global_tuning.range_width,
                nullptr,
                &input.match_mask);
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
