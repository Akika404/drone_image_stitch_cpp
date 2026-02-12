#pragma once

#include <string>

struct PosRecord {
    std::string time_str;
    double longitude;
    double latitude;
    double altitude;
    double roll;
    double pitch;
    double heading;
    std::string file_id;

    int timeSeconds() const;

    bool isValid() const;
};
