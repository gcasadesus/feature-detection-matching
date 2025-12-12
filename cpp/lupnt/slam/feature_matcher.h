
#pragma once

#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <string>

#include "lupnt/core/asset_factory.h"
#include "lupnt/core/config.h"
#include "lupnt/slam/feature_set.h"

namespace lupnt {

  struct Matches {
    MatX2i indexes;
    VecXd distances;

    int Size() const { return indexes.rows(); }
  };

  /**
   * @brief Abstract base class for feature matchers
   *
   * This class can be extended in Python to wrap neural network-based matchers
   * like SuperGlue, LightGlue, while providing OpenCV implementations in C++.
   */
  class FeatureMatcher {
  public:
    FeatureMatcher() = default;
    virtual ~FeatureMatcher() = default;

    /**
     * @brief Match features between two images
     * @param features1 Features from first image
     * @param features2 Features from aecond image
     * @return Vector of matches
     */
    virtual Matches Match(const Features& features1, const Features& features2) = 0;

    /**
     * @brief Get the name of the matcher
     */
    virtual std::string GetName() const = 0;

    static std::shared_ptr<FeatureMatcher> FromConfig(const Config& config);
  };

  // OpenCVFeatureMatcher
  class OpencvFeatureMatcher : public FeatureMatcher {
  public:
    OpencvFeatureMatcher() = default;
    ~OpencvFeatureMatcher() = default;

    Matches Match(const Features& features1, const Features& features2) override;

    std::string GetName() const override;

  protected:
    cv::Ptr<cv::DescriptorMatcher> matcher_;
    std::string name_;
  };

  // FlannBasedMatcher
  class FlannMatcher : public OpencvFeatureMatcher {
  public:
    FlannMatcher(Config& config);
    ~FlannMatcher() = default;
  };

  // BFMatcher
  class BruteForceMatcher : public OpencvFeatureMatcher {
  public:
    BruteForceMatcher(Config& config);
    ~BruteForceMatcher() = default;
  };

  // Declare the specialization before instantiation
  template <> std::unordered_map<std::string, AssetFactory<FeatureMatcher, Config&>::Creator>&
  AssetFactory<FeatureMatcher, Config&>::GetRegistry();

  // Ensure single registry instance across library boundaries
  extern template class AssetFactory<FeatureMatcher, Config&>;

}  // namespace lupnt
