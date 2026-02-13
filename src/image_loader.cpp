#include "image_loader.hpp"

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <utility>

#include <algorithm>

namespace fs = std::filesystem;

namespace {
std::string extract_image_id(const std::string &path) {
    fs::path p(path);
    std::string base = p.filename().string();
    auto pos = base.find('_');
    if (pos != std::string::npos) {
        return base.substr(0, pos);
    }
    auto dot = base.find_last_of('.');
    if (dot != std::string::npos) {
        return base.substr(0, dot);
    }
    return base;
}
}

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
        throw std::runtime_error("need at least 2 images to stitch");
    }

    std::vector<cv::Mat> imgs;
    for (auto &p: paths) {
        cv::Mat img = cv::imread(p);
        if (img.empty()) {
            std::cout << "read failed: " << p << std::endl;
            continue;
        }
        std::cout << "load: " << p << std::endl;
        imgs.push_back(img);
    }
    return imgs;
}

LoadedImages ImageLoader::loadWithIds(const std::string &folder) {
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

    if (paths.empty()) {
        throw std::runtime_error("no usable images found");
    }

    LoadedImages result;
    for (auto &p: paths) {
        cv::Mat img = cv::imread(p);
        if (img.empty()) {
            std::cout << "read failed: " << p << std::endl;
            continue;
        }
        std::cout << "load: " << p << std::endl;
        result.images.push_back(img);
        result.ids.push_back(extract_image_id(p));
    }
    return result;
}
