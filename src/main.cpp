
#include <algorithm>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>

#include "image_loader.hpp"
#include "pos_guided_stitcher.hpp"
#include "pos_reader.hpp"
#include "stitch_config.hpp"

namespace fs = std::filesystem;

namespace {
    constexpr double kEarthRadiusMeters = 6378137.0;

    struct CliOptions {
        std::string images_dir = "images/visible/full";
        std::string pos_path = "assets/pos.mti";
        std::string output_path = "output/visible/stitched_visible.jpg";
        std::string image_type = "visible";
        int max_images = 0;
        double compose_mpx = -1.0;
    };

    void printHelp() {
        std::cout << "Usage: drone_image_stitch_cpp [options]\n"
                << "Options:\n"
                << "  --images <dir>         Input image directory\n"
                << "  --pos <file>           POS/MTI file path\n"
                << "  --output <file>        Output stitched image path\n"
                << "  --image-type <type>    visible | nir | lwir\n"
                << "  --max-images <N>       Limit image count for quick tests\n"
                << "  --compose-mpx <value>  Composing resolution in mega-pixels per image\n"
                << "  --help                 Show this message\n";
    }

    bool parseArgs(const int argc, char **argv, CliOptions &opts) {
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            const auto needValue = [&](const std::string &key) {
                if (i + 1 >= argc) {
                    throw std::runtime_error("missing value for " + key);
                }
            };

            if (arg == "--images") {
                needValue(arg);
                opts.images_dir = argv[++i];
            } else if (arg == "--pos") {
                needValue(arg);
                opts.pos_path = argv[++i];
            } else if (arg == "--output") {
                needValue(arg);
                opts.output_path = argv[++i];
            } else if (arg == "--image-type") {
                needValue(arg);
                opts.image_type = argv[++i];
            } else if (arg == "--max-images") {
                needValue(arg);
                opts.max_images = std::max(0, std::stoi(argv[++i]));
            } else if (arg == "--compose-mpx") {
                needValue(arg);
                opts.compose_mpx = std::stod(argv[++i]);
            } else if (arg == "--help" || arg == "-h") {
                printHelp();
                return false;
            } else {
                throw std::runtime_error("unknown argument: " + arg);
            }
        }
        return true;
    }

    cv::Point2d lonLatToLocalMeters(const PosRecord &pos, const double lon0, const double lat0) {
        const double lat0_rad = lat0 * CV_PI / 180.0;
        const double x = (pos.longitude - lon0) * CV_PI / 180.0 * kEarthRadiusMeters * std::cos(lat0_rad);
        const double y = (pos.latitude - lat0) * CV_PI / 180.0 * kEarthRadiusMeters;
        return {x, y};
    }

    std::vector<StitchFrame> filterNoisyTurnFrames(const std::vector<StitchFrame> &frames) {
        if (frames.size() < 3) {
            return frames;
        }

        std::vector<cv::Point2d> xy;
        xy.reserve(frames.size());
        const double lon0 = frames.front().pos.longitude;
        const double lat0 = frames.front().pos.latitude;
        for (const auto &frame: frames) {
            xy.push_back(lonLatToLocalMeters(frame.pos, lon0, lat0));
        }

        std::vector<char> keep(frames.size(), 1);
        for (size_t i = 1; i + 1 < frames.size(); ++i) {
            const cv::Point2d v1 = xy[i] - xy[i - 1];
            const cv::Point2d v2 = xy[i + 1] - xy[i];
            const double d1 = cv::norm(v1);
            const double d2 = cv::norm(v2);

            if (d1 < 0.8 || d2 < 0.8) {
                keep[i] = 0;
                continue;
            }

            const double dot = v1.dot(v2) / (d1 * d2);
            const double clamped = std::clamp(dot, -1.0, 1.0);
            const double turn_deg = std::acos(clamped) * 180.0 / CV_PI;
            if (turn_deg > 45.0 && (d1 < 8.0 || d2 < 8.0)) {
                keep[i] = 0;
            }
        }

        std::vector<StitchFrame> out;
        out.reserve(frames.size());
        for (size_t i = 0; i < frames.size(); ++i) {
            if (keep[i]) {
                out.push_back(frames[i]);
            }
        }
        return out;
    }
}

int main(const int argc, char **argv) {
    try {
        CliOptions opts;
        if (!parseArgs(argc, argv, opts)) {
            return 0;
        }

        if (!fs::exists(opts.images_dir)) {
            throw std::runtime_error("image directory not found: " + opts.images_dir);
        }
        if (!fs::exists(opts.pos_path)) {
            throw std::runtime_error("POS file not found: " + opts.pos_path);
        }

        const LoadedImages image_index = ImageLoader::listWithIds(opts.images_dir);
        const std::vector<PosRecord> pos_records = PosReader::load(opts.pos_path);
        StitchTuning tuning = loadStitchTuning(opts.image_type);
        if (opts.compose_mpx > 0) {
            tuning.compositing_resol_mpx = opts.compose_mpx;
        }

        std::unordered_map<std::string, PosRecord> pos_by_id;
        pos_by_id.reserve(pos_records.size());
        for (const auto &record: pos_records) {
            if (!record.isValid()) {
                continue;
            }
            if (!pos_by_id.contains(record.file_id)) {
                pos_by_id[record.file_id] = record;
            }
        }

        std::vector<StitchFrame> frames;
        frames.reserve(image_index.ids.size());
        for (size_t i = 0; i < image_index.ids.size(); ++i) {
            const auto it = pos_by_id.find(image_index.ids[i]);
            if (it == pos_by_id.end()) {
                continue;
            }
            StitchFrame frame;
            frame.id = image_index.ids[i];
            frame.path = image_index.paths[i];
            frame.pos = it->second;
            frames.push_back(std::move(frame));
        }

        std::ranges::sort(frames, [](const StitchFrame &a, const StitchFrame &b) {
            const int ta = a.pos.timeSeconds();
            const int tb = b.pos.timeSeconds();
            if (ta != tb) {
                return ta < tb;
            }
            return a.id < b.id;
        });
        const size_t before_filter = frames.size();
        frames = filterNoisyTurnFrames(frames);

        if (opts.max_images > 0 && static_cast<int>(frames.size()) > opts.max_images) {
            frames.resize(opts.max_images);
        }

        std::cout << "[INPUT] images indexed: " << image_index.ids.size() << std::endl;
        std::cout << "[INPUT] valid POS records: " << pos_by_id.size() << std::endl;
        std::cout << "[INPUT] kept after turn/noise filter: " << frames.size()
                << "/" << before_filter << std::endl;
        std::cout << "[INPUT] matched frames: " << frames.size() << std::endl;
        std::cout << "[INPUT] output: " << opts.output_path << std::endl;

        const StitchOutput result = stitchWithPosGuidance(frames, tuning, opts.output_path);
        if (!result.ok) {
            std::cerr << "[ERROR] " << result.message << std::endl;
            return 2;
        }

        std::cout << "[DONE] " << result.message
                << ", stitched_images=" << result.stitched_count
                << ", file=" << result.output_path << std::endl;
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "[FATAL] " << e.what() << std::endl;
        return 1;
    }
}
