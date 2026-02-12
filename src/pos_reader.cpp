#include "pos_reader.hpp"

#include <fstream>
#include <iostream>
#include <sstream>

std::vector<PosRecord> PosReader::load(const std::string &pos_path) {
    std::ifstream file(pos_path);
    if (!file.is_open()) {
        throw std::runtime_error("POS文件不存在: " + pos_path);
    }

    std::vector<PosRecord> records;
    std::string line;
    int total_lines = 0;
    int skipped_lines = 0;

    std::cout << "[POS] 开始读取: " << pos_path << std::endl;
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
            record.roll = std::stod(parts[4]);
            record.pitch = std::stod(parts[5]);
            record.heading = std::stod(parts[6]);
            record.file_id = parts[7];
            records.push_back(record);
        } catch (...) {
            ++skipped_lines;
        }
    }

    std::cout << "[POS] 完成读取: records=" << records.size()
            << ", 总行数=" << total_lines << ", 跳过=" << skipped_lines << std::endl;
    return records;
}
