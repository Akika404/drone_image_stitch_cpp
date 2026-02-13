#include "pos_image_grouper.hpp"

#include <cmath>
#include <deque>
#include <filesystem>
#include <iostream>
#include <unordered_map>

std::vector<FlightStripGroup> PosBasedImageGrouper::groupWithRecords(
    const std::vector<cv::Mat> &images,
    const std::vector<std::string> &image_ids,
    const std::vector<PosRecord> &pos_records) {
    if (images.empty()) {
        std::cout << "[Group] empty images, return empty groups" << std::endl;
        return {};
    }
    if (image_ids.size() != images.size()) {
        throw std::runtime_error("image_ids must align with images");
    }

    std::cout << "[Group] grouping: images=" << images.size()
            << ", image_ids=" << image_ids.size()
            << ", pos_records=" << pos_records.size() << std::endl;

    std::unordered_map<std::string, std::deque<cv::Mat> > id_to_images;
    for (size_t i = 0; i < images.size(); ++i) {
        auto id = normalizeImageId(image_ids[i]);
        id_to_images[id].push_back(images[i]);
    }

    const auto strips = groupByFlightStrips(pos_records);
    std::vector<FlightStripGroup> groups;
    for (auto &strip: strips) {
        std::vector<PosRecord> sorted_records = strip;
        std::ranges::sort(sorted_records, [](const PosRecord &a, const PosRecord &b) {
            return a.timeSeconds() < b.timeSeconds();
        });
        FlightStripGroup group;
        for (auto &record: sorted_records) {
            auto it = id_to_images.find(record.file_id);
            if (it != id_to_images.end() && !it->second.empty()) {
                group.images.push_back(it->second.front());
                group.records.push_back(record);
                it->second.pop_front();
            }
        }
        if (!group.images.empty()) {
            groups.push_back(group);
        }
    }

    std::cout << "[Group] done: strips=" << strips.size()
            << ", valid_groups=" << groups.size() << std::endl;
    return groups;
}

std::vector<std::vector<cv::Mat> > PosBasedImageGrouper::group(
    const std::vector<cv::Mat> &images,
    const std::vector<std::string> &image_ids,
    const std::vector<PosRecord> &pos_records) const {
    const auto groups_with_records = groupWithRecords(images, image_ids, pos_records);
    std::vector<std::vector<cv::Mat> > groups;
    groups.reserve(groups_with_records.size());
    for (const auto &group: groups_with_records) {
        groups.push_back(group.images);
    }
    return groups;
}

auto PosBasedImageGrouper::groupByFlightStrips(
    const std::vector<PosRecord> &records,
    double heading_threshold,
    int min_strip_records,
    double stability_threshold,
    int stability_count) -> std::vector<std::vector<PosRecord> > {
    std::vector<PosRecord> valid_records;
    for (auto &r: records) {
        if (r.isValid()) valid_records.push_back(r);
    }
    if (valid_records.empty()) {
        std::cout << "[Group] POS records empty or all invalid" << std::endl;
        return {};
    }
    std::cout << "[Group] valid POS records: " << valid_records.size()
            << "/" << records.size() << std::endl;

    std::vector<std::vector<PosRecord> > strips;
    std::vector<PosRecord> current_strip_records;
    std::vector<PosRecord> stable_buffer;
    std::string state = "TURNING";

    for (size_t i = 0; i < valid_records.size(); ++i) {
        const auto &current_record = valid_records[i];
        if (state == "COLLECTING") {
            if (!current_strip_records.empty()) {
                std::vector<PosRecord> recent;
                size_t start = current_strip_records.size() > 10
                                   ? current_strip_records.size() - 10
                                   : 0;
                recent.insert(recent.end(),
                              current_strip_records.begin() + static_cast<long>(start),
                              current_strip_records.end());
                double avg_heading = stripAverageHeading(recent);
                double diff_from_avg = headingDifference(current_record.heading, avg_heading);
                const PosRecord *prev_record = i > 0 ? &valid_records[i - 1] : nullptr;
                double diff_from_prev = prev_record
                                            ? headingDifference(current_record.heading, prev_record->heading)
                                            : 0.0;
                if (diff_from_prev > heading_threshold || diff_from_avg > heading_threshold) {
                    if (static_cast<int>(current_strip_records.size()) >= min_strip_records) {
                        strips.push_back(current_strip_records);
                    }
                    state = "TURNING";
                    current_strip_records.clear();
                    stable_buffer.clear();
                } else {
                    current_strip_records.push_back(current_record);
                }
            } else {
                current_strip_records.push_back(current_record);
            }
        } else {
            if (stable_buffer.empty()) {
                stable_buffer.push_back(current_record);
            } else {
                const auto &last_in_buffer = stable_buffer.back();
                double diff = headingDifference(current_record.heading, last_in_buffer.heading);
                if (diff <= stability_threshold) {
                    stable_buffer.push_back(current_record);
                    if (static_cast<int>(stable_buffer.size()) >= stability_count) {
                        std::vector<double> buffer_headings;
                        buffer_headings.reserve(stable_buffer.size());
                        for (auto &r: stable_buffer) buffer_headings.push_back(r.heading);
                        double avg_buffer_heading = averageHeading(buffer_headings);
                        bool all_stable = true;
                        for (auto &h: buffer_headings) {
                            if (headingDifference(h, avg_buffer_heading) > stability_threshold) {
                                all_stable = false;
                                break;
                            }
                        }
                        if (all_stable) {
                            state = "COLLECTING";
                            current_strip_records = stable_buffer;
                            stable_buffer.clear();
                        }
                    }
                } else {
                    stable_buffer.clear();
                    stable_buffer.push_back(current_record);
                }
            }
        }
    }

    if (state == "COLLECTING" &&
        static_cast<int>(current_strip_records.size()) >= min_strip_records) {
        strips.push_back(current_strip_records);
    } else if (state == "TURNING" &&
               static_cast<int>(stable_buffer.size()) >= min_strip_records) {
        strips.push_back(stable_buffer);
    }
    return strips;
}

std::string PosBasedImageGrouper::normalizeImageId(const std::string &image_id) {
    std::filesystem::path p(image_id);
    std::string base = p.filename().string();
    if (const auto pos = base.find('_'); pos != std::string::npos) {
        return base.substr(0, pos);
    }
    if (const auto dot = base.find_last_of('.'); dot != std::string::npos) {
        return base.substr(0, dot);
    }
    return base;
}

double PosBasedImageGrouper::normalizeHeading(double heading) {
    while (heading > 180.0) heading -= 360.0;
    while (heading < -180.0) heading += 360.0;
    return heading;
}

double PosBasedImageGrouper::headingDifference(double h1, double h2) {
    h1 = normalizeHeading(h1);
    h2 = normalizeHeading(h2);
    double diff = std::abs(h1 - h2);
    if (diff > 180.0) diff = 360.0 - diff;
    return diff;
}

double PosBasedImageGrouper::averageHeading(const std::vector<double> &headings) {
    if (headings.empty()) return 0.0;
    double sin_sum = 0.0;
    double cos_sum = 0.0;
    for (auto &h: headings) {
        const double rad = h * CV_PI / 180.0;
        sin_sum += std::sin(rad);
        cos_sum += std::cos(rad);
    }
    return std::atan2(sin_sum, cos_sum) * 180.0 / CV_PI;
}

double PosBasedImageGrouper::stripAverageHeading(const std::vector<PosRecord> &records) {
    if (records.empty()) return 0.0;
    std::vector<double> headings;
    headings.reserve(records.size());
    for (auto &r: records) headings.push_back(r.heading);
    return averageHeading(headings);
}
