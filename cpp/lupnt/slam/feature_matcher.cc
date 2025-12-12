#include "lupnt/slam/feature_matcher.h"

#include <opencv2/features2d.hpp>

#include "lupnt/core/asset_factory.h"
#include "lupnt/core/logger.h"

namespace lupnt {

  std::shared_ptr<FeatureMatcher> FeatureMatcher::FromConfig(const Config& config) {
    auto loaded_config = LoadConfig(config);
    return AssetFactory<FeatureMatcher, Config&>::Create(loaded_config["class"].as<std::string>(),
                                                         loaded_config);
  }

  // OpenCVFeatureMatcher implementations
  Matches OpencvFeatureMatcher::Match(const Features& features1, const Features& features2) {
    // Check for empty features
    if (features1.Empty() || features2.Empty()) {
      Logger::Debug("Empty feature sets, returning zero matches", "OpencvFeatureMatcher");
      return Matches{MatX2i::Zero(0, 2), VecXd::Zero(0)};
    }

    auto desc1 = features1.descriptors;
    auto desc2 = features2.descriptors;

    Logger::Debug(
        fmt::format("Input descriptors: type1={}, type2={}, size1=({},{}), size2=({},{})",
                    desc1.type(), desc2.type(), desc1.rows, desc1.cols, desc2.rows, desc2.cols),
        "OpencvFeatureMatcher");

    // Use simple matching (no KNN, no ratio test)
    std::vector<cv::DMatch> matches;
    matcher_->match(desc1, desc2, matches);

    Logger::Debug(fmt::format("Found {} matches", matches.size()), "OpencvFeatureMatcher");

    MatX2i matches_mat(matches.size(), 2);
    VecXd distances(matches.size());
    for (size_t i = 0; i < matches.size(); ++i) {
      matches_mat(i, 0) = matches[i].queryIdx;
      matches_mat(i, 1) = matches[i].trainIdx;
      distances(i) = matches[i].distance;
    }
    return Matches{matches_mat, distances};
  }

  std::string OpencvFeatureMatcher::GetName() const { return name_; }

  FlannMatcher::FlannMatcher(Config& config) : OpencvFeatureMatcher() {
    name_ = "FlannBasedMatcher";

    // Read the *type* of index to use from the config
    // We'll default to "KDTree" if not specified
    std::string index_type = config["index_type"].as<std::string>("KDTree");

    cv::Ptr<cv::flann::IndexParams> index_params;

    if (index_type == "KDTree") {
      int trees = config["trees"].as<int>(5);
      index_params = cv::makePtr<cv::flann::KDTreeIndexParams>(trees);

      Logger::Debug(fmt::format("Created FlannMatcher with KDTreeIndexParams, trees={}", trees),
                    name_);

    } else if (index_type == "LSH") {
      int table_number = config["lsh_table_number"].as<int>(6);
      int key_size = config["lsh_key_size"].as<int>(12);
      int multi_probe_level = config["lsh_multi_probe_level"].as<int>(1);

      index_params
          = cv::makePtr<cv::flann::LshIndexParams>(table_number, key_size, multi_probe_level);

      Logger::Debug(fmt::format("Created FlannMatcher with LshIndexParams, table_number={}, "
                                "key_size={}, multi_probe_level={}",
                                table_number, key_size, multi_probe_level),
                    name_);
    } else {
      LUPNT_CHECK(false, "Unknown Flann index_type: " + index_type, name_);
    }

    int checks = config["checks"].as<int>(50);
    auto search_params = cv::makePtr<cv::flann::SearchParams>(checks);

    // Create the matcher with the chosen index
    matcher_ = cv::makePtr<cv::FlannBasedMatcher>(index_params, search_params);
  }

  BruteForceMatcher::BruteForceMatcher(Config& config) : OpencvFeatureMatcher() {
    name_ = "BFMatcher";

    // Parse norm_type as either string or int using magic_enum
    cv::NormTypes norm_type = cv::NORM_HAMMING;  // default for binary descriptors like ORB
    if (config["norm_type"].IsDefined()) {
      norm_type = ReadEnum<cv::NormTypes>(config["norm_type"]);
    }

    bool cross_check = config["cross_check"].as<bool>(false);
    matcher_ = cv::makePtr<cv::BFMatcher>(norm_type, cross_check);

    Logger::Debug(fmt::format("Created BruteForceMatcher with norm_type={}",
                              magic_enum::enum_name(norm_type)),
                  name_);
  }

  REGISTER_FACTORY_CLASS(FeatureMatcher, FlannMatcher);
  REGISTER_FACTORY_CLASS(FeatureMatcher, BruteForceMatcher);

  // Define the GetRegistry function for this specialization (must come before explicit
  // instantiation)
  template <> std::unordered_map<std::string, AssetFactory<FeatureMatcher, Config&>::Creator>&
  AssetFactory<FeatureMatcher, Config&>::GetRegistry() {
    return Registry();
  }

  // Explicitly instantiate the template to ensure single registry across library boundaries
  template class AssetFactory<FeatureMatcher, Config&>;

}  // namespace lupnt
