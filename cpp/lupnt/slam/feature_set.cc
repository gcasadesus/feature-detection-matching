#include "lupnt/slam/feature_set.h"

#include <algorithm>
#include <numeric>

namespace lupnt {

  Features::Features(const std::vector<cv::KeyPoint>& kpts, const cv::Mat& desc)
      : descriptors(desc) {
    if (kpts.empty()) return;

    uv = MatX2d(kpts.size(), 2);
    scores = VecXd(kpts.size());
    scales = VecXd(kpts.size());
    orientations = VecXd(kpts.size());

    for (size_t i = 0; i < kpts.size(); ++i) {
      uv(i, 0) = kpts[i].pt.x;
      uv(i, 1) = kpts[i].pt.y;
      scores(i) = kpts[i].response;
      scales(i) = kpts[i].size;
      orientations(i) = kpts[i].angle;
    }
  }

  void Features::FilterBest(size_t n) {
    if (n >= Size()) return;

    std::vector<size_t> indices(Size());
    std::iota(indices.begin(), indices.end(), 0);

    // Sort by scores if available, otherwise by keypoint response
    std::sort(indices.begin(), indices.end(),
              [this](size_t a, size_t b) { return scores(a) > scores(b); });

    // Keep only the first n indices (which are now sorted by score)
    indices.resize(n);

    cv::Mat desc(n, descriptors.cols, descriptors.type());
    for (size_t i = 0; i < n; ++i) descriptors.row(indices[i]).copyTo(desc.row(i));
    descriptors = std::move(desc);

    // Explicitly create new matrices to avoid any view/reference issues
    VecXd new_scores(n);
    VecXd new_scales(n);
    VecXd new_orientations(n);
    MatX2d new_uv(n, 2);

    for (size_t i = 0; i < n; ++i) {
      new_scores(i) = scores(indices[i]);
      new_scales(i) = scales(indices[i]);
      new_orientations(i) = orientations(indices[i]);
      new_uv(i, 0) = uv(indices[i], 0);
      new_uv(i, 1) = uv(indices[i], 1);
    }

    scores = std::move(new_scores);
    scales = std::move(new_scales);
    orientations = std::move(new_orientations);
    uv = std::move(new_uv);
  }

  void Features::Filter(const VecXi& indices) {
    cv::Mat desc(indices.size(), descriptors.cols, descriptors.type());
    for (size_t i = 0; i < indices.size(); ++i) descriptors.row(indices[i]).copyTo(desc.row(i));
    descriptors = std::move(desc);

    // Explicitly create new matrices to avoid any view/reference issues
    VecXd new_scores(indices.size());
    VecXd new_scales(indices.size());
    VecXd new_orientations(indices.size());
    MatX2d new_uv(indices.size(), 2);

    for (Eigen::Index i = 0; i < indices.size(); ++i) {
      new_scores(i) = scores(indices(i));
      new_scales(i) = scales(indices(i));
      new_orientations(i) = orientations(indices(i));
      new_uv(i, 0) = uv(indices(i), 0);
      new_uv(i, 1) = uv(indices(i), 1);
    }

    scores = std::move(new_scores);
    scales = std::move(new_scales);
    orientations = std::move(new_orientations);
    uv = std::move(new_uv);
  }
}  // namespace lupnt
