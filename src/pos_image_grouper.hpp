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
        double heading_threshold = 45.0,
        int min_strip_records = 5,
        double stability_threshold = 10.0,
        int stability_count = 5);

    /// Compute the "core" heading of a strip using the middle 50% of records
    /// (immune to edge contamination from turning photos).
    static double computeCoreHeading(const std::vector<PosRecord> &records);

    /// Remove turning photos from both ends of a strip.  Records whose heading
    /// deviates from the core heading by more than trim_threshold are dropped.
    static void trimStripEdges(
        std::vector<PosRecord> &records,
        double trim_threshold = 15.0);

    static std::string normalizeImageId(const std::string &image_id);

    static double normalizeHeading(double heading);

    static double headingDifference(double h1, double h2);

    static double averageHeading(const std::vector<double> &headings);

    static double stripAverageHeading(const std::vector<PosRecord> &records);

    /// Compute the flight axis (0°–180°) from a single strip's records using
    /// the double-angle circular averaging trick (folds 0° and 180° into the
    /// same axis).
    static double computeFlightAxis(const std::vector<PosRecord> &records);

    /// Overload: compute the dominant flight axis over multiple strips.
    static double computeFlightAxis(
        const std::vector<std::vector<PosRecord>> &strips);
};
