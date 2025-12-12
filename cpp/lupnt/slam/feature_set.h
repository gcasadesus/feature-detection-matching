#pragma once

#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <vector>

#include "lupnt/core/definitions.h"

namespace lupnt {

  /**
   * @brief Generic feature container that works fodr both traditional and learning-based methods
   *
   * Uses Eigen for numeric data and OpenCV for keypoints/images.
   * Optional fields allow different feature types to store only what they need.
   */
  struct Features {
    // Core features (always present)
    cv::Mat descriptors;  // [N x descriptor_dim] row-major
    MatX2d uv;            // [N x 2] pixel coordinates
    VecXd scores;         // [N] confidence scores
    VecXd scales;         // [N] scale information (for SIFT, etc.)
    VecXd orientations;   // [N] orientation angles (for SIFT, etc.)
    cv::Mat image;        // Source image

    Features() = default;

    Features(const std::vector<cv::KeyPoint>& kpts, const cv::Mat& desc);

    size_t Size() const { return uv.rows(); }
    bool Empty() const { return uv.rows() == 0; }
    void FilterBest(size_t n);
    void Filter(const VecXi& indices);
  };

}  // namespace lupnt
