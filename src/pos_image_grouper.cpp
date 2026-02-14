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

    auto strips = groupByFlightStrips(pos_records);

    // Trim turning photos from each strip's edges.
    for (size_t si = 0; si < strips.size(); si++) {
        const size_t before = strips[si].size();
        trimStripEdges(strips[si]);
        const size_t after = strips[si].size();
        if (before != after) {
            std::cout << "[Group] strip " << si << ": trimmed "
                    << (before - after) << " turning records ("
                    << before << " -> " << after << ")" << std::endl;
        }
    }

    // Remove strips that became too small after trimming.
    std::erase_if(strips, [](const std::vector<PosRecord> &s) { return s.size() < 3; });

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
    const std::vector<PosRecord> &pos_records) {
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
    // -----------------------------------------------------------------------
    // Step 1 — collect valid records
    // -----------------------------------------------------------------------
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

    // -----------------------------------------------------------------------
    // Step 2 — altitude filter: keep only records at survey altitude
    //          (>= 90% of max altitude). This removes ground operations,
    //          ascent, descent, and landing records.
    // -----------------------------------------------------------------------
    double max_alt = 0;
    for (auto &r: valid_records) max_alt = std::max(max_alt, r.altitude);
    const double alt_threshold = max_alt * 0.90;

    std::vector<PosRecord> survey;
    for (auto &r: valid_records) {
        if (r.altitude >= alt_threshold) survey.push_back(r);
    }
    std::cout << "[Group] altitude filter: " << survey.size() << "/"
            << valid_records.size() << " records (>= "
            << alt_threshold << "m, max=" << max_alt << "m)" << std::endl;

    if (survey.size() < static_cast<size_t>(min_strip_records)) {
        std::cout << "[Group] too few survey-altitude records" << std::endl;
        return {};
    }

    // -----------------------------------------------------------------------
    // Step 3 — compute GPS-based flight bearing for each record
    //          The POS kappa (κ) column is the camera orientation angle, NOT
    //          the flight compass direction.  Computing the bearing from
    //          consecutive GPS coordinates gives the actual flight direction,
    //          which is much more reliable for strip splitting.
    // -----------------------------------------------------------------------
    const double cos_lat = std::cos(survey[0].latitude * CV_PI / 180.0);
    for (size_t i = 0; i + 1 < survey.size(); i++) {
        const double dlat = survey[i + 1].latitude - survey[i].latitude;
        const double dlon = (survey[i + 1].longitude - survey[i].longitude) * cos_lat;
        const double dist = std::sqrt(dlat * dlat + dlon * dlon);
        if (dist > 1e-9) {
            survey[i].bearing = std::atan2(dlon, dlat) * 180.0 / CV_PI;
        }
        // else: drone is hovering; bearing stays 0 (default)
    }
    if (survey.size() > 1) {
        survey.back().bearing = survey[survey.size() - 2].bearing;
    }

    // -----------------------------------------------------------------------
    // Step 4 — state machine: split into strips by bearing changes
    // -----------------------------------------------------------------------
    std::vector<std::vector<PosRecord> > strips;
    std::vector<PosRecord> current_strip_records;
    std::vector<PosRecord> stable_buffer;
    std::string state = "TURNING";

    for (size_t i = 0; i < survey.size(); ++i) {
        const auto &current_record = survey[i];
        if (state == "COLLECTING") {
            if (!current_strip_records.empty()) {
                std::vector<PosRecord> recent;
                size_t start = current_strip_records.size() > 10
                                   ? current_strip_records.size() - 10
                                   : 0;
                recent.insert(recent.end(),
                              current_strip_records.begin() + static_cast<long>(start),
                              current_strip_records.end());
                double avg_bearing = stripAverageBearing(recent);
                double diff_from_avg = angleDifference(current_record.bearing, avg_bearing);
                const PosRecord *prev_record = i > 0 ? &survey[i - 1] : nullptr;
                double diff_from_prev = prev_record
                                            ? angleDifference(current_record.bearing, prev_record->bearing)
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
                double diff = angleDifference(current_record.bearing, last_in_buffer.bearing);
                if (diff <= stability_threshold) {
                    stable_buffer.push_back(current_record);
                    if (static_cast<int>(stable_buffer.size()) >= stability_count) {
                        std::vector<double> buffer_bearings;
                        buffer_bearings.reserve(stable_buffer.size());
                        for (auto &r: stable_buffer) buffer_bearings.push_back(r.bearing);
                        double avg_buffer_bearing = averageAngle(buffer_bearings);
                        bool all_stable = true;
                        for (auto &b: buffer_bearings) {
                            if (angleDifference(b, avg_buffer_bearing) > stability_threshold) {
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

    std::cout << "[Group] state machine found " << strips.size()
            << " candidate strips" << std::endl;
    for (size_t i = 0; i < strips.size(); i++) {
        const double axis = computeFlightAxis(strips[i]);
        std::cout << "[Group]   candidate " << i << ": " << strips[i].size()
                << " records, flight-axis=" << axis << "deg" << std::endl;
    }

    // -----------------------------------------------------------------------
    // Step 5 — flight axis alignment filter
    //          In a boustrophedon ("几") survey all strips share the same
    //          flight axis (just in opposite directions: N↔S or E↔W).
    //          Remove any strip whose axis doesn't align with the dominant
    //          survey axis (e.g. transit to/from the survey area, return).
    //          Only apply when there are > 2 candidates so we have a reliable
    //          majority to compute the axis from.
    // -----------------------------------------------------------------------
    if (strips.size() > 2) {
        const double survey_axis = computeFlightAxis(strips);
        std::cout << "[Group] dominant survey axis: " << survey_axis
                << "deg" << std::endl;

        // 这个地方用于过滤转弯时的部分，低于此数量的照片的分组会被舍弃掉
        constexpr double axis_tolerance = 20.0;
        std::vector<std::vector<PosRecord> > aligned;
        for (size_t si = 0; si < strips.size(); si++) {
            const double strip_axis = computeFlightAxis(strips[si]);
            double diff = std::abs(strip_axis - survey_axis);
            if (diff > 90.0) diff = 180.0 - diff;

            if (diff <= axis_tolerance) {
                aligned.push_back(strips[si]);
                std::cout << "[Group]   keep strip " << si << " ("
                        << strips[si].size() << " records, axis="
                        << strip_axis << "deg)" << std::endl;
            } else {
                std::cout << "[Group]   remove strip " << si << " ("
                        << strips[si].size() << " records, axis="
                        << strip_axis << "deg, diff=" << diff
                        << "deg from survey axis)" << std::endl;
            }
        }
        strips = aligned;
    }

    std::cout << "[Group] final strip count: " << strips.size() << std::endl;
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

double PosBasedImageGrouper::normalizeAngle(double angle) {
    while (angle > 180.0) angle -= 360.0;
    while (angle < -180.0) angle += 360.0;
    return angle;
}

double PosBasedImageGrouper::angleDifference(double a1, double a2) {
    a1 = normalizeAngle(a1);
    a2 = normalizeAngle(a2);
    double diff = std::abs(a1 - a2);
    if (diff > 180.0) diff = 360.0 - diff;
    return diff;
}

double PosBasedImageGrouper::averageAngle(const std::vector<double> &angles) {
    if (angles.empty()) return 0.0;
    double sin_sum = 0.0;
    double cos_sum = 0.0;
    for (auto &a: angles) {
        const double rad = a * CV_PI / 180.0;
        sin_sum += std::sin(rad);
        cos_sum += std::cos(rad);
    }
    return std::atan2(sin_sum, cos_sum) * 180.0 / CV_PI;
}

double PosBasedImageGrouper::stripAverageBearing(const std::vector<PosRecord> &records) {
    if (records.empty()) return 0.0;
    std::vector<double> bearings;
    bearings.reserve(records.size());
    for (auto &r: records) bearings.push_back(r.bearing);
    return averageAngle(bearings);
}

double PosBasedImageGrouper::computeFlightAxis(const std::vector<PosRecord> &records) {
    if (records.empty()) return 0.0;
    double sin2_sum = 0.0, cos2_sum = 0.0;
    for (auto &r: records) {
        const double rad = r.bearing * CV_PI / 180.0;
        sin2_sum += std::sin(2.0 * rad);
        cos2_sum += std::cos(2.0 * rad);
    }
    double axis = 0.5 * std::atan2(sin2_sum, cos2_sum) * 180.0 / CV_PI;
    if (axis < 0) axis += 180.0;
    return axis;
}

double PosBasedImageGrouper::computeFlightAxis(
    const std::vector<std::vector<PosRecord> > &strips) {
    if (strips.empty()) return 0.0;
    double sin2_sum = 0.0, cos2_sum = 0.0;
    for (auto &s: strips) {
        for (auto &r: s) {
            const double rad = r.bearing * CV_PI / 180.0;
            sin2_sum += std::sin(2.0 * rad);
            cos2_sum += std::cos(2.0 * rad);
        }
    }
    double axis = 0.5 * std::atan2(sin2_sum, cos2_sum) * 180.0 / CV_PI;
    if (axis < 0) axis += 180.0;
    return axis;
}

double PosBasedImageGrouper::computeCoreBearing(const std::vector<PosRecord> &records) {
    if (records.empty()) return 0.0;
    // Use the middle 50% of records — these are guaranteed to be straight-line
    // flight, uncontaminated by turning photos at the edges.
    const size_t n = records.size();
    const size_t start = n / 4;
    size_t end = start + n / 2;
    if (end > n) end = n;
    if (end <= start) return stripAverageBearing(records);

    std::vector<double> mid_bearings;
    mid_bearings.reserve(end - start);
    for (size_t i = start; i < end; i++) {
        mid_bearings.push_back(records[i].bearing);
    }
    return averageAngle(mid_bearings);
}

void PosBasedImageGrouper::trimStripEdges(
    std::vector<PosRecord> &records,
    const double trim_threshold) {
    if (records.size() < 6) return; // too short to trim safely

    const double core = computeCoreBearing(records);

    // -----------------------------------------------------------------------
    // Build a reference centre-line from the middle 50% of records.
    // These interior records are guaranteed to be stable straight-flight.
    // -----------------------------------------------------------------------
    const size_t n = records.size();
    const size_t q1 = n / 4;
    size_t q3 = q1 + n / 2;
    if (q3 > n) q3 = n;
    if (q3 <= q1 + 1) {
        // Fallback: strip too short for centre-line, use bearing only.
        while (records.size() > 3) {
            if (angleDifference(records.front().bearing, core) > trim_threshold)
                records.erase(records.begin());
            else break;
        }
        while (records.size() > 3) {
            if (angleDifference(records.back().bearing, core) > trim_threshold)
                records.pop_back();
            else break;
        }
        return;
    }

    // Approximate cos(latitude) for lon→metre-like conversion.
    const double cos_lat = std::cos(records[n / 2].latitude * CV_PI / 180.0);

    // Centre-line direction vector from q1 to q3-1 (interior endpoints).
    const double line_dx = (records[q3 - 1].longitude - records[q1].longitude) * cos_lat;
    const double line_dy = records[q3 - 1].latitude - records[q1].latitude;
    const double line_len = std::sqrt(line_dx * line_dx + line_dy * line_dy);

    // Anchor point on the centre-line (use q1).
    const double anchor_lon = records[q1].longitude;
    const double anchor_lat = records[q1].latitude;

    // Average along-track spacing in the interior, used to set a sensible
    // cross-track threshold that adapts to flight altitude / photo interval.
    double along_sum = 0;
    for (size_t i = q1; i + 1 < q3; i++) {
        const double adx = (records[i + 1].longitude - records[i].longitude) * cos_lat;
        const double ady = records[i + 1].latitude - records[i].latitude;
        along_sum += std::sqrt(adx * adx + ady * ady);
    }
    const double avg_spacing = (q3 - q1 > 1)
                                   ? along_sum / static_cast<double>(q3 - q1 - 1)
                                   : line_len;

    // Cross-track distance threshold: 1.5× average photo spacing.
    // This is generous enough to tolerate natural GPS drift in straight flight,
    // but catches even the first turning photo whose lateral offset is already
    // noticeable.
    const double ct_threshold = avg_spacing * 1.5;

    // Helper: compute cross-track distance (perpendicular to centre-line) for
    // a given record.  Returns 0 if the centre-line is degenerate.
    auto crossTrackDist = [&](const PosRecord &rec) -> double {
        if (line_len < 1e-12) return 0.0;
        const double px = (rec.longitude - anchor_lon) * cos_lat;
        const double py = rec.latitude - anchor_lat;
        // |cross product| / |line| = perpendicular distance
        return std::abs(px * line_dy - py * line_dx) / line_len;
    };

    // Trim from front: remove records deviating by bearing OR cross-track.
    while (records.size() > 3) {
        const bool bearing_bad = angleDifference(records.front().bearing, core) > trim_threshold;
        const bool position_bad = crossTrackDist(records.front()) > ct_threshold;
        if (bearing_bad || position_bad) {
            records.erase(records.begin());
        } else {
            break;
        }
    }

    // Trim from back.
    while (records.size() > 3) {
        const bool bearing_bad = angleDifference(records.back().bearing, core) > trim_threshold;
        const bool position_bad = crossTrackDist(records.back()) > ct_threshold;
        if (bearing_bad || position_bad) {
            records.pop_back();
        } else {
            break;
        }
    }
}
