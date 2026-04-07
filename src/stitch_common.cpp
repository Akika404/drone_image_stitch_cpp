#include "stitch_common.hpp"


void autoCropBlackBorder(cv::Mat &pano) {
    cv::Mat gray;
    cv::cvtColor(pano, gray, cv::COLOR_BGR2GRAY);

    cv::Mat thresh;
    cv::threshold(gray, thresh, 1, 255, cv::THRESH_BINARY);

    std::vector<std::vector<cv::Point> > contours;
    cv::findContours(thresh, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if (contours.empty()) {
        return;
    }

    cv::Rect max_rect = cv::boundingRect(contours[0]);
    double max_area = cv::contourArea(contours[0]);
    for (size_t i = 1; i < contours.size(); ++i) {
        const double area = cv::contourArea(contours[i]);
        if (area > max_area) {
            max_area = area;
            max_rect = cv::boundingRect(contours[i]);
        }
    }
    pano = pano(max_rect).clone();
}

std::string stitchStatusToString(const cv::Stitcher::Status status) {
    switch (status) {
        case cv::Stitcher::OK:
            return "OK";
        case cv::Stitcher::ERR_NEED_MORE_IMGS:
            return "need more images";
        case cv::Stitcher::ERR_HOMOGRAPHY_EST_FAIL:
            return "homography estimation failed";
        case cv::Stitcher::ERR_CAMERA_PARAMS_ADJUST_FAIL:
            return "camera params adjust failed";
        default:
            return "unknown error";
    }
}
