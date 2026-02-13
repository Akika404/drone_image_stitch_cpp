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

    [[nodiscard]] static std::vector<std::vector<cv::Mat> > group(
        const std::vector<cv::Mat> &images,
        const std::vector<std::string> &image_ids,
        const std::vector<PosRecord> &pos_records);

private:
    static std::vector<std::vector<PosRecord> > groupByFlightStrips(
        const std::vector<PosRecord> &records,
        double heading_threshold = 45.0,
        int min_strip_records = 5,
        double stability_threshold = 10.0,
        int stability_count = 5);

    /// Compute the "core" bearing of a strip using the middle 50% of records
    /// (immune to edge contamination from turning photos).
    static double computeCoreBearing(const std::vector<PosRecord> &records);

    /// Remove turning photos from both ends of a strip.  Records whose bearing
    /// deviates from the core bearing by more than trim_threshold are dropped.
    static void trimStripEdges(
        std::vector<PosRecord> &records,
        double trim_threshold = 15.0);

    static std::string normalizeImageId(const std::string &image_id);

    /// Normalize any angle to [-180°, 180°].
    static double normalizeAngle(double angle);

    /// Shortest angular distance between two angles (0°–180°).
    static double angleDifference(double a1, double a2);

    /// Circular average of a list of angles (degrees).
    static double averageAngle(const std::vector<double> &angles);

    /// Average bearing of records in a strip (uses record.bearing).
    static double stripAverageBearing(const std::vector<PosRecord> &records);

    /// Compute the flight axis (0°–180°) from a single strip's records using
    /// the double-angle circular averaging trick (folds 0° and 180° into the
    /// same axis).
    static double computeFlightAxis(const std::vector<PosRecord> &records);

    /// Overload: compute the dominant flight axis over multiple strips.
    static double computeFlightAxis(
        const std::vector<std::vector<PosRecord> > &strips);
};
