#include <opencv2/opencv.hpp>
#include <opencv2/stitching.hpp>
#include <opencv2/stitching/detail/matchers.hpp>

#include <filesystem>
#include <iostream>

#include "image_loader.hpp"

namespace fs = std::filesystem;
using namespace cv;
using namespace std;

namespace {
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
} // namespace

int main() {
    constexpr string image_folder = "../images";
    constexpr string image_type = "visible";
    constexpr string group = "group_1";

    string input_folder = image_folder + "/" + image_type + "/" + group;
    string output_folder = "../output/" + image_type + "/" + group;
    fs::create_directories(output_folder);
    fs::path output_path = fs::path(output_folder) / "uav_panorama_sift.jpg";

    try {
        cout << "[Main] 输入目录: " << input_folder << endl;
        cout << "[Main] 输出文件: " << output_path << endl;

        vector<Mat> images = ImageLoader::load(input_folder);
        cout << "[Main] 有效图像数量: " << images.size() << endl;

        cout << "[Main] 创建 OpenCV Stitcher(SCANS) + SIFT..." << endl;
        Ptr<Stitcher> stitcher = Stitcher::create(Stitcher::SCANS);

        // 可替换为 ORB / AKAZE 等
        stitcher->setFeaturesFinder(SIFT::create(1200));
        stitcher->setFeaturesMatcher(makePtr<detail::BestOf2NearestMatcher>(false, 0.4f));

        cout << "[Main] 开始拼接" << endl;
        Mat panorama;
        Stitcher::Status status = stitcher->stitch(images, panorama);
        if (status != Stitcher::OK) {
            throw runtime_error(
                "拼接失败: " + stitchStatusToString(status) + " (错误码: " + to_string(status) + ")");
        }
        cout << "[Main] 拼接成功，开始自动裁剪黑边..." << endl;

        autoCropBlackBorder(panorama);

        imwrite(output_path.string(), panorama);
        cout << "[Finish] 拼接完成: " << output_path << endl;

        imshow("Panorama", panorama);
        waitKey(0);
        destroyAllWindows();
    } catch (const exception &e) {
        cerr << "[Error] 错误: " << e.what() << endl;
        return 1;
    }

    return 0;
}
