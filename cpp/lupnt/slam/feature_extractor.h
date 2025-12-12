#pragma once

#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <string>

#include "lupnt/core/asset_factory.h"
#include "lupnt/core/config.h"
#include "lupnt/slam/feature_set.h"

namespace lupnt {

  /**
   * @brief Abstract base class for feature extractors
   *
   * This class can be extended in Python to wrap neural network-based extractors
   * like SuperPoint, while providing OpenCV implementations in C++.
   */
  class FeatureExtractor {
  public:
    FeatureExtractor() = default;
    virtual ~FeatureExtractor() = default;

    /**
     * @brief Extract features from an image
     * @param image Input image (grayscale or color)
     * @return Features containing keypoints, descriptors, and optional metadata
     */
    virtual Features Extract(const cv::Mat& image) = 0;

    /**
     * @brief Get the name of the extractor
     */
    virtual std::string GetName() const = 0;

    static std::shared_ptr<FeatureExtractor> FromConfig(const Config& config);
  };

  class OpencvFeatureExtractor : public FeatureExtractor {
  public:
    OpencvFeatureExtractor() = default;
    ~OpencvFeatureExtractor() = default;

    Features Extract(const cv::Mat& image) override;
    std::string GetName() const override { return name_; }
    cv::Ptr<cv::Feature2D> GetDetector() const { return detector_; }

  protected:
    cv::Ptr<cv::Feature2D> detector_;
    std::string name_;
  };

  class Sift : public OpencvFeatureExtractor {
  public:
    Sift(Config& config);
    ~Sift() = default;
  };

  class Orb : public OpencvFeatureExtractor {
  public:
    Orb(Config& config);
    ~Orb() = default;
  };

  class Brisk : public OpencvFeatureExtractor {
  public:
    Brisk(Config& config);
    ~Brisk() = default;
  };

  class Akaze : public OpencvFeatureExtractor {
  public:
    Akaze(Config& config);
    ~Akaze() = default;
  };

  // Declare the specialization before instantiation
  template <> std::unordered_map<std::string, AssetFactory<FeatureExtractor, Config&>::Creator>&
  AssetFactory<FeatureExtractor, Config&>::GetRegistry();

  // Ensure single registry instance across library boundaries
  extern template class AssetFactory<FeatureExtractor, Config&>;

}  // namespace lupnt
