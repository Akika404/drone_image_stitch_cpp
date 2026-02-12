#pragma once

#include <string>
#include <vector>

#include "pos_record.hpp"

class PosReader {
public:
    static std::vector<PosRecord> load(const std::string &pos_path);
};
