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

double degreeToRadian(const double degree) {
    return degree * CV_PI / 180.0;
}

double averageFlightAxisHeading(const std::vector<PosRecord> &records) {
    if (records.empty()) {
        return 0.0;
    }
    double sin_sum = 0.0;
    double cos_sum = 0.0;
    for (const auto &record: records) {
        const double angle = degreeToRadian(record.bearing);
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

std::pair<double, double> stripCentroidLonLat(const std::vector<PosRecord> &records) {
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
