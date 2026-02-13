#pragma once

#include <opencv2/opencv.hpp>

#include <utility>
#include <vector>

#include "pos_record.hpp"

void autoCropBlackBorder(cv::Mat &pano);

std::string stitchStatusToString(cv::Stitcher::Status status);

double degreeToRadian(double degree);

double averageFlightAxisHeading(const std::vector<PosRecord> &records);

std::pair<double, double> stripCentroidLonLat(const std::vector<PosRecord> &records);
