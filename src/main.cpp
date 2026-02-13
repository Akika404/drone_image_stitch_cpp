#include <opencv2/core/ocl.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/stitching.hpp>
#include <opencv2/stitching/detail/blenders.hpp>
#include <opencv2/stitching/detail/exposure_compensate.hpp>
#include <opencv2/stitching/detail/matchers.hpp>
#include <opencv2/stitching/detail/motion_estimators.hpp>
#include <opencv2/stitching/detail/seam_finders.hpp>
#include <opencv2/stitching/warpers.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "image_loader.hpp"
#include "pos_image_grouper.hpp"
#include "pos_reader.hpp"

namespace fs = std::filesystem;
using namespace cv;
using namespace std;

namespace {
struct StripPanorama {
    Mat pano;
    vector<PosRecord> records;
    double cross_track = 0.0;
};

struct StitchTuning {
    int sift_features = 2200;
    float match_conf = 0.35f;
    int min_good_matches = 10;
    int min_inliers = 8;

    bool use_range_matcher = true;
    int range_width = 6;

    bool use_affine_bundle = true;
    bool use_affine_warper = true;
    bool use_blocks_gain = true;
    int blend_bands = 5;
    float pano_conf_thresh = 0.7f;
};

struct PairDiagnostics {
    size_t kp_left = 0;
    size_t kp_right = 0;
    size_t good_matches = 0;
    bool descriptors_ready = false;
    bool homography_ok = false;
    int inliers = 0;
};

bool envEnabled(const char *name, const bool default_value = false) {
    const char *value = std::getenv(name);
    if (!value) {
        return default_value;
    }
    const string text(value);
    if (text == "1" || text == "true" || text == "TRUE" || text == "on" || text == "ON") {
        return true;
    }
    if (text == "0" || text == "false" || text == "FALSE" || text == "off" || text == "OFF") {
        return false;
    }
    return default_value;
}

int envIntOr(const char *name, const int default_value, const int min_value) {
    const char *value = std::getenv(name);
    if (!value) {
        return default_value;
    }
    char *end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value || *end != '\0') {
        return default_value;
    }
    if (parsed < min_value) {
        return min_value;
    }
    return static_cast<int>(parsed);
}

float envFloatOr(const char *name, const float default_value, const float min_value) {
    const char *value = std::getenv(name);
    if (!value) {
        return default_value;
    }
    char *end = nullptr;
    const float parsed = std::strtof(value, &end);
    if (end == value || *end != '\0') {
        return default_value;
    }
    if (parsed < min_value) {
        return min_value;
    }
    return parsed;
}

StitchTuning loadStitchTuning() {
    StitchTuning tuning;
    tuning.sift_features = envIntOr("STITCH_SIFT_FEATURES", tuning.sift_features, 128);
    tuning.match_conf = envFloatOr("STITCH_MATCH_CONF", tuning.match_conf, 0.01f);
    tuning.min_good_matches = envIntOr("STITCH_MIN_GOOD_MATCHES", tuning.min_good_matches, 4);
    tuning.min_inliers = envIntOr("STITCH_MIN_INLIERS", tuning.min_inliers, 4);

    tuning.use_range_matcher = envEnabled("STITCH_USE_RANGE_MATCHER", tuning.use_range_matcher);
    tuning.range_width = envIntOr("STITCH_RANGE_WIDTH", tuning.range_width, 2);

    tuning.use_affine_bundle = envEnabled("STITCH_USE_AFFINE_BUNDLE", tuning.use_affine_bundle);
    tuning.use_affine_warper = envEnabled("STITCH_USE_AFFINE_WARPER", tuning.use_affine_warper);
    tuning.use_blocks_gain = envEnabled("STITCH_USE_BLOCKS_GAIN", tuning.use_blocks_gain);
    tuning.blend_bands = envIntOr("STITCH_BLEND_BANDS", tuning.blend_bands, 1);
    tuning.pano_conf_thresh = envFloatOr("STITCH_PANO_CONF", tuning.pano_conf_thresh, 0.01f);
    return tuning;
}

string matInfo(const Mat &img) {
    ostringstream oss;
    oss << img.cols << "x" << img.rows << ", ch=" << img.channels();
    return oss.str();
}

void autoCropBlackBorder(Mat &pano) {
    Mat gray;
    cvtColor(pano, gray, COLOR_BGR2GRAY);

    Mat thresh;
    threshold(gray, thresh, 1, 255, THRESH_BINARY);

    vector<vector<Point>> contours;
    findContours(thresh, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    if (contours.empty()) {
        return;
    }

    Rect max_rect = boundingRect(contours[0]);
    double max_area = contourArea(contours[0]);
    for (size_t i = 1; i < contours.size(); ++i) {
        const double area = contourArea(contours[i]);
        if (area > max_area) {
            max_area = area;
            max_rect = boundingRect(contours[i]);
        }
    }
    pano = pano(max_rect).clone();
}

string stitchStatusToString(const Stitcher::Status status) {
    switch (status) {
        case Stitcher::OK:
            return "OK";
        case Stitcher::ERR_NEED_MORE_IMGS:
            return "need more images";
        case Stitcher::ERR_HOMOGRAPHY_EST_FAIL:
            return "homography estimation failed";
        case Stitcher::ERR_CAMERA_PARAMS_ADJUST_FAIL:
            return "camera params adjust failed";
        default:
            return "unknown error";
    }
}

double degreeToRadian(const double degree) {
    return degree * CV_PI / 180.0;
}

double averageFlightAxisHeading(const vector<PosRecord> &records) {
    if (records.empty()) {
        return 0.0;
    }
    double sin_sum = 0.0;
    double cos_sum = 0.0;
    for (const auto &record : records) {
        const double angle = degreeToRadian(record.heading);
        sin_sum += std::sin(2.0 * angle);
        cos_sum += std::cos(2.0 * angle);
    }
    const double axis = 0.5 * std::atan2(sin_sum, cos_sum);
    double axis_deg = axis * 180.0 / CV_PI;
    if (axis_deg < 0.0) {
        axis_deg += 180.0;
    }
    return axis_deg;
}

pair<double, double> stripCentroidLonLat(const vector<PosRecord> &records) {
    if (records.empty()) {
        return {0.0, 0.0};
    }
    double lon_sum = 0.0;
    double lat_sum = 0.0;
    for (const auto &r : records) {
        lon_sum += r.longitude;
        lat_sum += r.latitude;
    }
    return {
        lon_sum / static_cast<double>(records.size()),
        lat_sum / static_cast<double>(records.size())
    };
}

void computeCrossTrackOrder(vector<StripPanorama> &strip_panos) {
    if (strip_panos.empty()) {
        return;
    }
    vector<PosRecord> all_records;
    for (const auto &strip : strip_panos) {
        all_records.insert(all_records.end(), strip.records.begin(), strip.records.end());
    }
    const double axis_heading_deg = averageFlightAxisHeading(all_records);
    const double axis_heading_rad = degreeToRadian(axis_heading_deg);

    const auto [base_lon, base_lat] = stripCentroidLonLat(strip_panos.front().records);
    const double base_lat_rad = degreeToRadian(base_lat);
    constexpr double meters_per_lat = 110540.0;
    const double meters_per_lon = 111320.0 * std::cos(base_lat_rad);

    const double along_east = std::sin(axis_heading_rad);
    const double along_north = std::cos(axis_heading_rad);
    const double cross_east = -along_north;
    const double cross_north = along_east;

    for (auto &strip : strip_panos) {
        const auto [lon, lat] = stripCentroidLonLat(strip.records);
        const double east = (lon - base_lon) * meters_per_lon;
        const double north = (lat - base_lat) * meters_per_lat;
        strip.cross_track = east * cross_east + north * cross_north;
    }

    ranges::sort(strip_panos, [](const StripPanorama &a, const StripPanorama &b) {
        return a.cross_track < b.cross_track;
    });
}

PairDiagnostics computePairDiagnostics(const Mat &left, const Mat &right, const int sift_features) {
    PairDiagnostics diag;
    Mat left_gray;
    Mat right_gray;
    if (left.channels() == 1) {
        left_gray = left;
    } else {
        cvtColor(left, left_gray, COLOR_BGR2GRAY);
    }
    if (right.channels() == 1) {
        right_gray = right;
    } else {
        cvtColor(right, right_gray, COLOR_BGR2GRAY);
    }

    Ptr<SIFT> sift = SIFT::create(sift_features);
    vector<KeyPoint> kp_left;
    vector<KeyPoint> kp_right;
    Mat desc_left;
    Mat desc_right;
    sift->detectAndCompute(left_gray, noArray(), kp_left, desc_left);
    sift->detectAndCompute(right_gray, noArray(), kp_right, desc_right);
    diag.kp_left = kp_left.size();
    diag.kp_right = kp_right.size();

    if (desc_left.empty() || desc_right.empty()) {
        return diag;
    }
    diag.descriptors_ready = true;

    BFMatcher matcher(NORM_L2);
    vector<vector<DMatch>> knn_matches;
    matcher.knnMatch(desc_left, desc_right, knn_matches, 2);
    vector<DMatch> good_matches;
    good_matches.reserve(knn_matches.size());
    for (const auto &m : knn_matches) {
        if (m.size() < 2) {
            continue;
        }
        if (m[0].distance < 0.75f * m[1].distance) {
            good_matches.push_back(m[0]);
        }
    }
    diag.good_matches = good_matches.size();

    if (good_matches.size() < 4) {
        return diag;
    }

    vector<Point2f> pts_left;
    vector<Point2f> pts_right;
    pts_left.reserve(good_matches.size());
    pts_right.reserve(good_matches.size());
    for (const auto &m : good_matches) {
        pts_left.push_back(kp_left[m.queryIdx].pt);
        pts_right.push_back(kp_right[m.trainIdx].pt);
    }

    Mat inlier_mask;
    Mat H = findHomography(pts_left, pts_right, RANSAC, 3.0, inlier_mask);
    if (H.empty()) {
        return diag;
    }
    diag.homography_ok = true;
    diag.inliers = countNonZero(inlier_mask);
    return diag;
}

void logPairDiagnostics(
    const Mat &left,
    const Mat &right,
    const string &stage,
    const size_t idx,
    const PairDiagnostics &diag,
    const StitchTuning &tuning) {
    cout << "[" << stage << "] failure diagnostics idx=" << idx
         << ", left={" << matInfo(left) << "}, right={" << matInfo(right) << "}"
         << ", kp_left=" << diag.kp_left
         << ", kp_right=" << diag.kp_right;
    if (!diag.descriptors_ready) {
        cout << ", desc_empty=true" << endl;
        return;
    }
    cout << ", good_matches=" << diag.good_matches
         << "(min=" << tuning.min_good_matches << ")";
    if (!diag.homography_ok) {
        if (diag.good_matches < 4) {
            cout << ", homography=not_enough_matches" << endl;
        } else {
            cout << ", homography=failed" << endl;
        }
        return;
    }
    cout << ", homography=inliers/good_matches="
         << diag.inliers << "/" << diag.good_matches
         << "(min=" << tuning.min_inliers << ")" << endl;
}

Ptr<Stitcher> createConfiguredStitcher(
    const Stitcher::Mode mode,
    const StitchTuning &tuning,
    const int range_width_override = -1) {
    Ptr<Stitcher> stitcher = Stitcher::create(mode);

    stitcher->setPanoConfidenceThresh(tuning.pano_conf_thresh);
    stitcher->setWaveCorrection(false);

    stitcher->setFeaturesFinder(SIFT::create(tuning.sift_features));

    const int range_width = (range_width_override > 0) ? range_width_override : tuning.range_width;
    if (tuning.use_range_matcher && range_width > 1) {
        stitcher->setFeaturesMatcher(makePtr<detail::BestOf2NearestRangeMatcher>(range_width, false, tuning.match_conf));
    } else {
        stitcher->setFeaturesMatcher(makePtr<detail::BestOf2NearestMatcher>(false, tuning.match_conf));
    }

    if (tuning.use_affine_bundle) {
        stitcher->setBundleAdjuster(makePtr<detail::BundleAdjusterAffinePartial>());
    }

    if (tuning.use_affine_warper) {
        stitcher->setWarper(makePtr<AffineWarper>());
    }

    stitcher->setSeamFinder(makePtr<detail::DpSeamFinder>(detail::DpSeamFinder::COLOR_GRAD));

    if (tuning.use_blocks_gain) {
        stitcher->setExposureCompensator(makePtr<detail::BlocksGainCompensator>());
    }

    stitcher->setBlender(makePtr<detail::MultiBandBlender>(false, tuning.blend_bands));
    return stitcher;
}

Stitcher::Status stitchWithMode(
    const vector<Mat> &images,
    Mat &output,
    const Stitcher::Mode mode,
    const string &stage,
    const StitchTuning &tuning,
    const int range_width_override = -1) {
    if (images.empty()) {
        return Stitcher::ERR_NEED_MORE_IMGS;
    }
    if (images.size() == 1) {
        output = images.front().clone();
        return Stitcher::OK;
    }

    if (images.size() == 2) {
        const auto diag = computePairDiagnostics(images[0], images[1], tuning.sift_features);
        if (!diag.descriptors_ready || static_cast<int>(diag.good_matches) < tuning.min_good_matches) {
            logPairDiagnostics(images[0], images[1], stage, 1, diag, tuning);
            return Stitcher::ERR_HOMOGRAPHY_EST_FAIL;
        }
        if (!diag.homography_ok || diag.inliers < tuning.min_inliers) {
            logPairDiagnostics(images[0], images[1], stage, 1, diag, tuning);
            return Stitcher::ERR_HOMOGRAPHY_EST_FAIL;
        }
    }

    Ptr<Stitcher> stitcher = createConfiguredStitcher(mode, tuning, range_width_override);
    return stitcher->stitch(images, output);
}

optional<Mat> stitchSequentially(
    const vector<Mat> &images,
    const Stitcher::Mode mode,
    const string &stage_name,
    const StitchTuning &tuning,
    const int range_width_override = -1) {
    if (images.empty()) {
        return nullopt;
    }
    Mat current = images.front().clone();
    for (size_t i = 1; i < images.size(); ++i) {
        Mat next_result;
        const vector<Mat> pair = {current, images[i]};
        const auto status = stitchWithMode(pair, next_result, mode, stage_name, tuning, range_width_override);
        if (status != Stitcher::OK) {
            const auto diag = computePairDiagnostics(current, images[i], tuning.sift_features);
            logPairDiagnostics(current, images[i], stage_name, i, diag, tuning);
            return nullopt;
        }
        current = next_result;
    }
    return current;
}

Mat stitchRobustly(
    const vector<Mat> &images,
    const Stitcher::Mode mode,
    const string &stage_name,
    const StitchTuning &tuning,
    const int range_width_override = -1) {
    Mat output;
    const auto first_try_status = stitchWithMode(images, output, mode, stage_name, tuning, range_width_override);
    if (first_try_status == Stitcher::OK) {
        return output;
    }

    cout << "[" << stage_name << "] one-shot stitch failed, fallback to sequential: "
         << stitchStatusToString(first_try_status) << endl;
    const auto sequential = stitchSequentially(images, mode, stage_name, tuning, range_width_override);
    if (sequential.has_value()) {
        return sequential.value();
    }

    throw runtime_error(
        "[" + stage_name + "] stitch failed: " + stitchStatusToString(first_try_status) +
        " (code: " + to_string(first_try_status) + ")");
}
} // namespace

int main() {
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);
    cv::ocl::setUseOpenCL(false);
    const auto temp_opencv = fs::temp_directory_path() / "opencv";
    fs::create_directories(temp_opencv);
    const string image_folder = "../images";
    const string image_type = "visible";
    const string group = "desert_1";
    const string pos_path = "../assets/pos.mti";

    const bool use_pos = envEnabled("STITCH_USE_POS", false);

    const string input_folder = image_folder + "/" + image_type + "/" + group;
    const string output_folder = "../output/" + image_type + "/" + group;
    fs::create_directories(output_folder);

    const string filename = image_type + "_" + group + "_" + "uav_panorama.jpg";
    const fs::path output_path = fs::path(output_folder) / filename;
    const fs::path output_dir = output_path.parent_path();
    const fs::path strip_dir = output_dir / "strips";
    fs::create_directories(strip_dir);

    const StitchTuning tuning = loadStitchTuning();

    try {
        cout << "[Main] input dir: " << input_folder << endl;
        cout << "[Main] POS: " << (use_pos ? "enabled" : "disabled") << endl;
        cout << "[Main] output: " << output_path << endl;
        cout << "[Main] stitch params: sift=" << tuning.sift_features
             << ", match_conf=" << tuning.match_conf
             << ", range_matcher=" << (tuning.use_range_matcher ? "on" : "off")
             << ", range_width=" << tuning.range_width
             << ", affine_bundle=" << (tuning.use_affine_bundle ? "on" : "off")
             << ", affine_warper=" << (tuning.use_affine_warper ? "on" : "off") << endl;

        const auto [images, ids] = ImageLoader::loadWithIds(input_folder);
        cout << "[Main] valid images: " << images.size() << endl;
        if (images.size() < 2) {
            throw runtime_error("need at least 2 images to stitch");
        }

        vector<FlightStripGroup> strip_groups;
        if (use_pos) {
            cout << "[Main] POS file: " << pos_path << endl;
            const auto pos_records = PosReader::load(pos_path);
            strip_groups = PosBasedImageGrouper::groupWithRecords(images, ids, pos_records);
            if (strip_groups.empty()) {
                throw runtime_error("POS grouping empty, check POS file and image IDs");
            }
        } else {
            strip_groups.push_back({images, {}});
            cout << "[Main] POS disabled, all images as single group" << endl;
        }

        vector<StripPanorama> strip_panos;
        for (size_t i = 0; i < strip_groups.size(); ++i) {
            if (strip_groups[i].images.size() < 2) {
                cout << "[Strip " << i << "] < 2 images, skip" << endl;
                continue;
            }
            cout << "[Strip " << i << "] intra-strip stitch, images=" << strip_groups[i].images.size() << endl;

            Mat strip_pano = stitchRobustly(
                strip_groups[i].images,
                Stitcher::SCANS,
                "Strip " + to_string(i),
                tuning,
                tuning.range_width);
            autoCropBlackBorder(strip_pano);

            const fs::path strip_output = strip_dir / ("strip_" + to_string(i) + ".jpg");
            imwrite(strip_output.string(), strip_pano);
            cout << "[Strip " << i << "] strip saved: " << strip_output << endl;

            strip_panos.push_back({strip_pano, strip_groups[i].records, 0.0});
        }

        if (strip_panos.empty()) {
            throw runtime_error("all strip stitches failed, cannot produce panorama");
        }

        Mat panorama;
        if (strip_panos.size() == 1) {
            panorama = strip_panos.front().pano.clone();
        } else {
            if (use_pos) {
                computeCrossTrackOrder(strip_panos);
            }

            vector<Mat> strip_images;
            strip_images.reserve(strip_panos.size());
            for (const auto &strip : strip_panos) {
                strip_images.push_back(strip.pano);
            }

            const int global_range = std::max(2, std::min(3, static_cast<int>(strip_images.size())));
            cout << "[Main] inter-strip stitch, strip_count=" << strip_images.size()
                 << ", global_range=" << global_range << endl;
            panorama = stitchRobustly(strip_images, Stitcher::SCANS, "Global", tuning, global_range);
        }

        autoCropBlackBorder(panorama);
        imwrite(output_path.string(), panorama);
        cout << "[Finish] done: " << output_path << endl;
    } catch (const exception &e) {
        cerr << "[Error] " << e.what() << endl;
        return 1;
    }

    return 0;
}
