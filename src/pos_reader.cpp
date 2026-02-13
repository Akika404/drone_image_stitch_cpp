#include "pos_reader.hpp"

#include <fstream>
#include <iostream>
#include <sstream>

std::vector<PosRecord> PosReader::load(const std::string &pos_path) {
    std::ifstream file(pos_path);
    if (!file.is_open()) {
        throw std::runtime_error("POS file not found: " + pos_path);
    }

    std::vector<PosRecord> records;
    std::string line;
    int total_lines = 0;
    int skipped_lines = 0;

    std::cout << "[POS] reading: " << pos_path << std::endl;
    while (std::getline(file, line)) {
        ++total_lines;
        if (line.empty()) {
            ++skipped_lines;
            continue;
        }
        std::istringstream iss(line);
        std::vector<std::string> parts;
        std::string part;
        while (iss >> part) {
            parts.push_back(part);
        }
        if (parts.size() < 8) {
            ++skipped_lines;
            continue;
        }
        try {
            PosRecord record;
            record.time_str = parts[0];
            record.longitude = std::stod(parts[1]);
            record.latitude = std::stod(parts[2]);
            record.altitude = std::stod(parts[3]);
            record.omega = std::stod(parts[4]);
            record.phi = std::stod(parts[5]);
            record.kappa = std::stod(parts[6]);
            record.file_id = parts[7];
            records.push_back(record);
        } catch (...) {
            ++skipped_lines;
        }
    }

    std::cout << "[POS] done: records=" << records.size()
            << ", total_lines=" << total_lines << ", skipped=" << skipped_lines << std::endl;
    return records;
}
