#include "image_loader.hpp"

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <utility>

#include <algorithm>

namespace fs = std::filesystem;

std::vector<cv::Mat> ImageLoader::load(const std::string &folder) {
    const std::vector<std::string> exts = {"jpg", "jpeg", "png", "bmp", "tiff"};
    std::vector<std::string> paths;

    for (auto &p: fs::directory_iterator(folder)) {
        if (!p.is_regular_file()) continue;
        std::string ext = p.path().extension().string();
        if (!ext.empty() && ext[0] == '.') ext = ext.substr(1);
        std::ranges::transform(ext, ext.begin(), ::tolower);
        for (auto &e: exts) {
            if (ext == e) paths.push_back(p.path().string());
        }
    }

    std::ranges::sort(paths);

    if (paths.size() < 2) {
        throw std::runtime_error("至少需要两张图像进行拼接");
    }

    std::vector<cv::Mat> imgs;
    for (auto &p: paths) {
        cv::Mat img = cv::imread(p);
        if (img.empty()) {
            std::cout << "读取失败: " << p << std::endl;
            continue;
        }
        std::cout << "加载: " << p << std::endl;
        imgs.push_back(img);
    }
    return imgs;
}
