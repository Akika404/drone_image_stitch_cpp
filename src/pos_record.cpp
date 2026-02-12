#include "pos_record.hpp"

#include <cmath>

int PosRecord::timeSeconds() const {
    if (time_str.size() == 6) {
        const int h = std::stoi(time_str.substr(0, 2));
        const int m = std::stoi(time_str.substr(2, 2));
        const int s = std::stoi(time_str.substr(4, 2));
        return h * 3600 + m * 60 + s;
    }
    return 0;
}

bool PosRecord::isValid() const {
    if (std::abs(longitude) < 1.0 || std::abs(latitude) < 1.0) {
        return false;
    }
    if (altitude < 50.0) {
        return false;
    }
    return true;
}
