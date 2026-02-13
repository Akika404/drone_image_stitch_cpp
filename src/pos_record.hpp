#pragma once

#include <string>

struct PosRecord {
    std::string time_str;       // 拍照时间 (HHMMSS)
    double longitude = 0;       // 经度 (WGS84, degrees)
    double latitude  = 0;       // 纬度 (WGS84, degrees)
    double altitude  = 0;       // 高度 (m)
    double omega     = 0;       // ω — 相机外方位元素旋转角 X (degrees)
    double phi       = 0;       // φ — 相机外方位元素旋转角 Y (degrees)
    double kappa     = 0;       // κ — 相机外方位元素旋转角 Z (degrees)
    std::string file_id;        // 照片编号

    /// GPS-based flight bearing (degrees, computed from consecutive coordinates).
    /// NOT from the POS file — filled by the grouping algorithm.
    /// 0° = North, 90° = East, ±180° = South, -90° = West.
    double bearing = 0;

    [[nodiscard]] int timeSeconds() const;

    [[nodiscard]] bool isValid() const;
};
