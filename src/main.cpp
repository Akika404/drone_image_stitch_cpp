#include <opencv2/opencv.hpp>

#include <filesystem>
#include <iostream>
#include <thread>

#include "image_loader.hpp"
#include "pos_image_grouper.hpp"
#include "pos_reader.hpp"
#include "stitch_pipeline.hpp"

namespace fs = std::filesystem;
using namespace cv;
using namespace cv::detail;
using namespace std;
using namespace uav;

StitchPipelineConfig createDefaultStitchPipelineConfig() {
    StitchPipelineConfig config;
    config.featureDetector = SIFT::create(1200);
    config.matcher = makePtr<BestOf2NearestMatcher>(false, 0.4f);
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
    string input_folder = "../images/visible/full";
    string output_folder = "../output/visible/full";
    string pos_path = "../assets/pos.mti";

    try {
        setNumThreads(static_cast<int>(std::thread::hardware_concurrency()));
        cout << "[Main] 输入目录: " << input_folder << endl;
        cout << "[Main] 输出目录: " << output_folder << endl;

        auto [images, ids] = ImageLoader::loadWithIds(input_folder);
        cout << "[Main] 有效图像数量: " << images.size() << endl;

        auto undistortImage = [](const Mat &input) {
            return input;
        };
        for (auto &img: images) {
            img = undistortImage(img);
        }

        vector<PosRecord> pos_records = PosReader::load(pos_path);
        PosBasedImageGrouper grouper;
        vector<vector<Mat> > groups = grouper.group(images, ids, pos_records);
        if (groups.empty()) {
            throw std::runtime_error("未找到有效航带分组");
        }

        StitchPipelineConfig config = createDefaultStitchPipelineConfig();
        StitchPipeline pipeline(config);

        vector<Mat> strip_panoramas;
        for (size_t i = 0; i < groups.size(); ++i) {
            auto &strip_images = groups[i];
            if (strip_images.empty()) continue;
            if (strip_images.size() == 1) {
                strip_panoramas.push_back(strip_images.front());
                continue;
            }
            cout << "[Main] 开始拼接航带 " << i << "，图像数: " << strip_images.size() << endl;
            Mat strip_panorama = pipeline.stitch(strip_images);
            cout << "[Main] 航带 " << i << " 拼接完成" << endl;
            strip_panoramas.push_back(strip_panorama);
        }

        if (strip_panoramas.empty()) {
            throw std::runtime_error("航带拼接失败，未生成有效图像");
        }

        Mat panorama;
        if (strip_panoramas.size() == 1) {
            panorama = strip_panoramas.front();
        } else {
            cout << "[Main] 开始拼接航带结果，数量: " << strip_panoramas.size() << endl;
            panorama = pipeline.stitch(strip_panoramas);
            cout << "[Main] 航带结果拼接完成" << endl;
        }

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
