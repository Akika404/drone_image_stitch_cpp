#include <opencv2/opencv.hpp>

#include <filesystem>
#include <iostream>

#include "stitch_pipeline.hpp"

namespace fs = std::filesystem;
using namespace cv;
using namespace cv::detail;
using namespace std;
using namespace uav;

vector<Mat> loadImages(const string &folder) {
    const vector<string> exts = {"jpg", "jpeg", "png", "bmp", "tiff"};
    vector<string> paths;

    for (auto &p: fs::directory_iterator(folder)) {
        if (!p.is_regular_file()) continue;
        string ext = p.path().extension().string();
        if (!ext.empty() && ext[0] == '.') ext = ext.substr(1);
        ranges::transform(ext, ext.begin(), ::tolower);
        for (auto &e: exts) {
            if (ext == e) paths.push_back(p.path().string());
        }
    }

    ranges::sort(paths);

    if (paths.size() < 2) {
        throw runtime_error("至少需要两张图像进行拼接");
    }

    vector<Mat> imgs;
    for (auto &p: paths) {
        Mat img = imread(p);
        if (img.empty()) {
            cout << "读取失败: " << p << endl;
            continue;
        }
        cout << "加载: " << p << endl;
        imgs.push_back(img);
    }
    return imgs;
}

StitchPipelineConfig createDefaultStitchPipelineConfig() {
    StitchPipelineConfig config;
    config.featureDetector = SIFT::create(2000);
    config.matcher = makePtr<BestOf2NearestMatcher>(false, 0.3f);
    config.estimator = makePtr<HomographyBasedEstimator>();
    config.bundleAdjuster = makePtr<BundleAdjusterRay>();
    config.warperCreator = makePtr<cv::PlaneWarper>();
    config.seamFinder = makePtr<DpSeamFinder>(DpSeamFinder::COLOR);
    config.exposureCompensator =
            ExposureCompensator::createDefault(ExposureCompensator::GAIN);
    config.blender = makePtr<MultiBandBlender>();
    return config;
}

int main() {
    string input_folder = "../images/visible/image_2";
    string output_folder = "../output/visible/image_2";

    try {
        cout << "[Main] 输入目录: " << input_folder << endl;
        cout << "[Main] 输出目录: " << output_folder << endl;

        vector<Mat> images = loadImages(input_folder);
        cout << "[Main] 有效图像数量: " << images.size() << endl;

        StitchPipelineConfig config = createDefaultStitchPipelineConfig();
        StitchPipeline pipeline(config);

        cout << "[Main] 开始执行拼接 Pipeline" << endl;
        Mat panorama = pipeline.stitch(images);
        cout << "[Main] 拼接 Pipeline 执行完成" << endl;

        fs::create_directories(output_folder);
        string filename = "uav_panorama.jpg";
        fs::path save_path = fs::path(output_folder) / filename;

        imwrite(save_path.string(), panorama);

        cout << "[Finish] 拼接完成: " << save_path << endl;

        imshow("Panorama", panorama);
        waitKey(0);
    } catch (exception &e) {
        cerr << "[Error] 错误: " << e.what() << endl;
    }

    return 0;
}
