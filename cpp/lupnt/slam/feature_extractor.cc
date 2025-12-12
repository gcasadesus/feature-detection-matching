#include "lupnt/slam/feature_extractor.h"

#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>

#include "lupnt/core/asset_factory.h"
#include "lupnt/core/logger.h"

namespace lupnt {

  std::shared_ptr<FeatureExtractor> FeatureExtractor::FromConfig(const Config& config) {
    auto loaded_config = LoadConfig(config);
    return AssetFactory<FeatureExtractor, Config&>::Create(loaded_config["class"].as<std::string>(),
                                                           loaded_config);
  }

  Features OpencvFeatureExtractor::Extract(const cv::Mat& image) {
    Logger::Debug(
        fmt::format("Extracting features: input image size=({},{}), channels={}, depth={}",
                    image.rows, image.cols, image.channels(), image.depth()),
        "OpencvFeatureExtractor");

    // Convert to grayscale if needed
    cv::Mat gray;
    if (image.channels() == 3) {
      cv::cvtColor(image, gray, cv::COLOR_RGB2GRAY);
      Logger::Debug("Converted RGB to grayscale", "OpencvFeatureExtractor");
    } else {
      gray = image;
    }

    // Convert to 8-bit unsigned if needed (OpenCV feature detectors require CV_8U)
    if (gray.depth() != CV_8U) {
      Logger::Debug(fmt::format("Converting image depth from {} to CV_8U", gray.depth()),
                    "OpencvFeatureExtractor");
      cv::Mat gray_8u;
      if (gray.depth() == CV_32F || gray.depth() == CV_64F) {
        // Assume float images are in range [0, 1], scale to [0, 255]
        double minVal, maxVal;
        cv::minMaxLoc(gray, &minVal, &maxVal);
        if (maxVal <= 1.0) {
          gray.convertTo(gray_8u, CV_8U, 255.0);
        } else {
          gray.convertTo(gray_8u, CV_8U);
        }
      } else {
        gray.convertTo(gray_8u, CV_8U);
      }
      gray = gray_8u;
    }

    // Detect and compute
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    detector_->detectAndCompute(gray, cv::noArray(), keypoints, descriptors);

    Logger::Debug(
        fmt::format("Detected {} keypoints, descriptor type={}, size=({},{})", keypoints.size(),
                    descriptors.type(), descriptors.rows, descriptors.cols),
        "OpencvFeatureExtractor");

    // Create Features
    Features features(keypoints, descriptors);
    features.image = gray;
    return features;
  }

  Sift::Sift(Config& config) : OpencvFeatureExtractor() {
    name_ = "SIFT";

    int n_features = config["n_features"].as<int>(0);
    int n_octave_layers = config["n_octave_layers"].as<int>(3);
    float contrast_threshold = config["contrast_threshold"].as<float>(0.04);
    float edge_threshold = config["edge_threshold"].as<float>(10.0);
    float sigma = config["sigma"].as<float>(1.6);

    detector_
        = cv::SIFT::create(n_features, n_octave_layers, contrast_threshold, edge_threshold, sigma);

    Logger::Debug(
        fmt::format("Created SIFT feature extractor with n_features={}, n_octave_layers={}, "
                    "contrast_threshold={}, edge_threshold={}, sigma={}",
                    n_features, n_octave_layers, contrast_threshold, edge_threshold, sigma),
        "Sift");
  }

  Orb::Orb(Config& config) : OpencvFeatureExtractor() {
    name_ = "ORB";

    int n_features = config["n_features"].as<int>(500);
    int n_levels = config["n_levels"].as<int>(8);
    float scale_factor = config["scale_factor"].as<float>(1.2f);
    int edge_threshold = config["edge_threshold"].as<int>(31);
    int first_level = config["first_level"].as<int>(0);
    int WTA_K = config["WTA_K"].as<int>(2);

    // Parse score_type as either string or int using magic_enum
    cv::ORB::ScoreType score_type = cv::ORB::HARRIS_SCORE;  // default
    if (config["score_type"].IsDefined()) {
      try {
        score_type = enum_value<cv::ORB::ScoreType>(config["score_type"].as<int>());
      } catch (const YAML::BadConversion&) {
        score_type = enum_cast<cv::ORB::ScoreType>(config["score_type"].as<std::string>())
                         .value_or(cv::ORB::HARRIS_SCORE);
      }
    }

    int patch_size = config["patch_size"].as<int>(31);
    int fast_threshold = config["fast_threshold"].as<int>(20);

    detector_ = cv::ORB::create(n_features, scale_factor, n_levels, edge_threshold, first_level,
                                WTA_K, score_type, patch_size, fast_threshold);

    Logger::Debug(
        fmt::format("Created ORB feature extractor with n_features={}, n_levels={}, "
                    "scale_factor={}, edge_threshold={}, "
                    "first_level={}, WTA_K={}, score_type={}, patch_size={}, fast_threshold={}",
                    n_features, n_levels, scale_factor, edge_threshold, first_level, WTA_K,
                    enum_name(score_type), patch_size, fast_threshold),
        "Orb");
  }

  Brisk::Brisk(Config& config) : OpencvFeatureExtractor() {
    name_ = "BRISK";

    int thresh = config["thresh"].as<int>(30);
    int octaves = config["octaves"].as<int>(3);
    float pattern_scale = config["pattern_scale"].as<float>(1.0f);

    detector_ = cv::BRISK::create(thresh, octaves, pattern_scale);

    Logger::Debug(
        fmt::format("Created BRISK feature extractor with thresh={}, octaves={}, pattern_scale={}",
                    thresh, octaves, pattern_scale),
        "Brisk");
  }

  Akaze::Akaze(Config& config) : OpencvFeatureExtractor() {
    name_ = "AKAZE";

    // Parse descriptor_type as either string or int using magic_enum
    cv::AKAZE::DescriptorType descriptor_type = cv::AKAZE::DESCRIPTOR_MLDB;  // default
    if (config["descriptor_type"].IsDefined()) {
      descriptor_type = ReadEnum<cv::AKAZE::DescriptorType>(config["descriptor_type"]);
    }

    int descriptor_size = config["descriptor_size"].as<int>(0);
    int descriptor_channels = config["descriptor_channels"].as<int>(3);
    float threshold = config["threshold"].as<float>(0.001f);
    int nOctaves = config["n_octaves"].as<int>(4);
    int nOctaveLayers = config["n_octave_layers"].as<int>(4);

    // Parse diffusivity as either string or int using magic_enum
    cv::KAZE::DiffusivityType diffusivity = cv::KAZE::DIFF_PM_G2;  // default
    if (config["diffusivity"].IsDefined()) {
      diffusivity = ReadEnum<cv::KAZE::DiffusivityType>(config["diffusivity"]);
    }

    int max_points = config["max_points"].as<int>(-1);

    detector_ = cv::AKAZE::create(descriptor_type, descriptor_size, descriptor_channels, threshold,
                                  nOctaves, nOctaveLayers, diffusivity, max_points);

    Logger::Debug(
        fmt::format("Created AKAZE feature extractor with descriptor_type={}, descriptor_size={}, "
                    "descriptor_channels={}, threshold={}, nOctaves={}, nOctaveLayers={}, "
                    "diffusivity={}, max_points={}",
                    enum_name(descriptor_type), descriptor_size, descriptor_channels, threshold,
                    nOctaves, nOctaveLayers, enum_name(diffusivity), max_points),
        "Akaze");
  }

  REGISTER_FACTORY_CLASS(FeatureExtractor, Sift);
  REGISTER_FACTORY_CLASS(FeatureExtractor, Orb);
  REGISTER_FACTORY_CLASS(FeatureExtractor, Akaze);
  REGISTER_FACTORY_CLASS(FeatureExtractor, Brisk);

  // Define the GetRegistry function for this specialization (must come before explicit
  // instantiation)
  template <> std::unordered_map<std::string, AssetFactory<FeatureExtractor, Config&>::Creator>&
  AssetFactory<FeatureExtractor, Config&>::GetRegistry() {
    return Registry();
  }

  // Explicitly instantiate the template to ensure single registry across library boundaries
  template class AssetFactory<FeatureExtractor, Config&>;

}  // namespace lupnt
