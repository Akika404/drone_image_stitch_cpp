#include <opencv2/opencv.hpp>
#include <opencv2/stitching.hpp>
#include <opencv2/stitching/detail/matchers.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <optional>
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

    void autoCropBlackBorder(Mat &pano) {
        Mat gray;
        cvtColor(pano, gray, COLOR_BGR2GRAY);

        Mat thresh;
        threshold(gray, thresh, 1, 255, THRESH_BINARY);

        vector<vector<Point> > contours;
        findContours(thresh, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        if (contours.empty()) {
            return;
        }

        Rect max_rect = boundingRect(contours[0]);
        double max_area = contourArea(contours[0]);
        for (size_t i = 1; i < contours.size(); ++i) {
            double area = contourArea(contours[i]);
            if (area > max_area) {
                max_area = area;
                max_rect = boundingRect(contours[i]);
            }
        }
        pano = pano(max_rect).clone();
    }

    string stitchStatusToString(Stitcher::Status status) {
        switch (status) {
            case Stitcher::OK:
                return "OK";
            case Stitcher::ERR_NEED_MORE_IMGS:
                return "需要更多图像";
            case Stitcher::ERR_HOMOGRAPHY_EST_FAIL:
                return "单应性矩阵估计失败";
            case Stitcher::ERR_CAMERA_PARAMS_ADJUST_FAIL:
                return "相机参数调整失败";
            default:
                return "未知错误";
        }
    }

    fs::path resolveExistingPath(const vector<fs::path> &candidates) {
        for (const auto &candidate: candidates) {
            if (fs::exists(candidate)) {
                return candidate;
            }
        }
        return candidates.front();
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
        for (const auto &record: records) {
            const double angle = degreeToRadian(record.heading);
            sin_sum += std::sin(2.0 * angle);
            cos_sum += std::cos(2.0 * angle);
        }
        const double axis = 0.5 * std::atan2(sin_sum, cos_sum);
        double axis_deg = axis * 180.0 / CV_PI;
        if (axis_deg < 0.0) axis_deg += 180.0;
        return axis_deg;
    }

    pair<double, double> stripCentroidLonLat(const vector<PosRecord> &records) {
        if (records.empty()) {
            return {0.0, 0.0};
        }
        double lon_sum = 0.0;
        double lat_sum = 0.0;
        for (const auto &r: records) {
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
        for (const auto &strip: strip_panos) {
            all_records.insert(all_records.end(), strip.records.begin(), strip.records.end());
        }
        const double axis_heading_deg = averageFlightAxisHeading(all_records);
        const double axis_heading_rad = degreeToRadian(axis_heading_deg);

        const auto [base_lon, base_lat] = stripCentroidLonLat(strip_panos.front().records);
        const double base_lat_rad = degreeToRadian(base_lat);
        constexpr double meters_per_lat = 110540.0;
        const double meters_per_lon = 111320.0 * std::cos(base_lat_rad);

        // heading定义: 北向为0°, 顺时针增加; 在东北坐标系上的方向向量为(sin, cos)
        const double along_east = std::sin(axis_heading_rad);
        const double along_north = std::cos(axis_heading_rad);
        const double cross_east = -along_north;
        const double cross_north = along_east;

        for (auto &strip: strip_panos) {
            const auto [lon, lat] = stripCentroidLonLat(strip.records);
            const double east = (lon - base_lon) * meters_per_lon;
            const double north = (lat - base_lat) * meters_per_lat;
            strip.cross_track = east * cross_east + north * cross_north;
        }

        ranges::sort(strip_panos, [](const StripPanorama &a, const StripPanorama &b) {
            return a.cross_track < b.cross_track;
        });
    }

    Stitcher::Status stitchWithMode(
        const vector<Mat> &images, Mat &output, Stitcher::Mode mode, const string &stage) {
        if (images.empty()) {
            return Stitcher::ERR_NEED_MORE_IMGS;
        }
        if (images.size() == 1) {
            output = images.front().clone();
            return Stitcher::OK;
        }
        cout << "[" << stage << "] 创建 Stitcher, mode="
                << (mode == Stitcher::SCANS ? "SCANS" : "PANORAMA")
                << ", image_count=" << images.size() << endl;

        Ptr<Stitcher> stitcher = Stitcher::create(mode);
        stitcher->setFeaturesFinder(SIFT::create(1500));
        stitcher->setFeaturesMatcher(makePtr<detail::BestOf2NearestMatcher>(false, 0.35f));

        return stitcher->stitch(images, output);
    }

    optional<Mat> stitchSequentially(
        const vector<Mat> &images,
        const Stitcher::Mode mode,
        const string &stage_name) {
        if (images.empty()) {
            return nullopt;
        }
        Mat current = images.front().clone();
        for (size_t i = 1; i < images.size(); ++i) {
            Mat next_result;
            const vector pair = {current, images[i]};
            const auto status = stitchWithMode(pair, next_result, mode, stage_name);
            if (status != Stitcher::OK) {
                cout << "[" << stage_name << "] 序贯拼接失败: idx=" << i
                        << ", status=" << stitchStatusToString(status) << endl;
                return nullopt;
            }
            current = next_result;
        }
        return current;
    }

    Mat stitchRobustly(const vector<Mat> &images, const Stitcher::Mode mode, const string &stage_name) {
        Mat output;
        const auto first_try_status = stitchWithMode(images, output, mode, stage_name);
        if (first_try_status == Stitcher::OK) {
            return output;
        }

        cout << "[" << stage_name << "] 一次性拼接失败，改为序贯拼接: "
                << stitchStatusToString(first_try_status) << endl;
        const auto sequential = stitchSequentially(images, mode, stage_name);
        if (sequential.has_value()) {
            return sequential.value();
        }

        throw runtime_error(
            "[" + stage_name + "] 拼接失败: " + stitchStatusToString(first_try_status) +
            " (错误码: " + to_string(first_try_status) + ")");
    }
} // namespace

int main() {
    const string input_folder = resolveExistingPath({
        "output/visible/full/strips",
        "../output/visible/full/strips"
    }).string();

    const string pos_path = resolveExistingPath({
        "assets/pos.mti",
        "../assets/pos.mti"
    }).string();

    const string output_path_str = (
        resolveExistingPath({"output", "../output"}) /
        "visible/full/uav_panorama_full_1.jpg").string();

    const fs::path output_path(output_path_str);
    const fs::path output_dir = output_path.parent_path();
    const fs::path strip_dir = output_dir / "strips";
    fs::create_directories(strip_dir);

    try {
        cout << "[Main] 输入目录: " << input_folder << endl;
        cout << "[Main] POS文件: " << pos_path << endl;
        cout << "[Main] 输出文件: " << output_path << endl;

        const auto [images, ids] = ImageLoader::loadWithIds(input_folder);
        cout << "[Main] 有效图像数量: " << images.size() << endl;
        const auto pos_records = PosReader::load(pos_path);

        PosBasedImageGrouper grouper;
        auto strip_groups = grouper.groupWithRecords(images, ids, pos_records);
        if (strip_groups.empty()) {
            throw runtime_error("POS分组结果为空，请检查POS文件与图片ID是否匹配");
        }

        vector<StripPanorama> strip_panos;
        for (size_t i = 0; i < strip_groups.size(); ++i) {
            if (strip_groups[i].images.size() < 2) {
                cout << "[Strip " << i << "] 图片少于2张，跳过" << endl;
                continue;
            }
            cout << "[Strip " << i << "] 开始航带内拼接, images="
                    << strip_groups[i].images.size() << endl;
            Mat strip_pano = stitchRobustly(
                strip_groups[i].images, Stitcher::SCANS, "Strip " + to_string(i));
            autoCropBlackBorder(strip_pano);

            const fs::path strip_output = strip_dir / ("strip_" + to_string(i) + ".jpg");
            imwrite(strip_output.string(), strip_pano);
            cout << "[Strip " << i << "] 航带图已保存: " << strip_output << endl;

            strip_panos.push_back({strip_pano, strip_groups[i].records, 0.0});
        }

        if (strip_panos.empty()) {
            throw runtime_error("所有航带拼接均失败，无法生成整图");
        }
        if (strip_panos.size() == 1) {
            Mat panorama = strip_panos.front().pano.clone();
            autoCropBlackBorder(panorama);
            imwrite(output_path.string(), panorama);
            cout << "[Finish] 仅检测到1条航带，输出完成: " << output_path << endl;
            return 0;
        }

        computeCrossTrackOrder(strip_panos);
        vector<Mat> strip_images;
        strip_images.reserve(strip_panos.size());
        for (const auto &strip: strip_panos) {
            strip_images.push_back(strip.pano);
        }

        cout << "[Main] 开始航带间二次拼接, strip_count=" << strip_images.size() << endl;
        Mat panorama = stitchRobustly(strip_images, Stitcher::SCANS, "Global");
        autoCropBlackBorder(panorama);
        imwrite(output_path.string(), panorama);
        cout << "[Finish] 拼接完成: " << output_path << endl;
    } catch (const exception &e) {
        cerr << "[Error] 错误: " << e.what() << endl;
        return 1;
    }

    return 0;
}
