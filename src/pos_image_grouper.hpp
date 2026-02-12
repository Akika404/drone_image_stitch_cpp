#pragma once

#include <opencv2/opencv.hpp>

#include <string>
#include <vector>

#include "pos_record.hpp"

struct FlightStripGroup {
    std::vector<cv::Mat> images;
    std::vector<PosRecord> records;
};

class PosBasedImageGrouper {
public:
    [[nodiscard]] static std::vector<FlightStripGroup> groupWithRecords(
        const std::vector<cv::Mat> &images,
        const std::vector<std::string> &image_ids,
        const std::vector<PosRecord> &pos_records);

    [[nodiscard]] std::vector<std::vector<cv::Mat> > group(
        const std::vector<cv::Mat> &images,
        const std::vector<std::string> &image_ids,
        const std::vector<PosRecord> &pos_records) const;

private:
    static std::vector<std::vector<PosRecord> > groupByFlightStrips(
        const std::vector<PosRecord> &records,
        double heading_threshold = 60.0,
        int min_strip_records = 5,
        double stability_threshold = 10.0,
        int stability_count = 3);

    static std::string normalizeImageId(const std::string &image_id);

    static double normalizeHeading(double heading);

    static double headingDifference(double h1, double h2);

    static double averageHeading(const std::vector<double> &headings);

    static double stripAverageHeading(const std::vector<PosRecord> &records);
};
