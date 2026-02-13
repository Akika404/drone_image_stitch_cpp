#include "pos_record.hpp"

#include <cmath>
#include <iostream>

int PosRecord::timeSeconds() const {
    if (time_str.size() == 6) {
        const int h = std::stoi(time_str.substr(0, 2));
        const int m = std::stoi(time_str.substr(2, 2));
        const int s = std::stoi(time_str.substr(4, 2));
        return h * 3600 + m * 60 + s;
    }
    return 0;
}

// 飞行高度限制，低于这个高度的数据将会被舍弃
constexpr int flt_height_limit = 50;

bool PosRecord::isValid() const {
    // 这里是为了防止出现缺一列的情况，时间未初始化的情况下，第一列会被读为经纬度
    // 经纬度一定是带有小数点的，所以用“是否包含‘.’”来判断
    if (time_str.find('.') != std::string::npos) {
        return false;
    }
    if (std::abs(longitude) < 1.0 || std::abs(latitude) < 1.0) {
        return false;
    }
    // 这个是飞行高度
    if (altitude < flt_height_limit) {
        return false;
    }
    return true;
}
