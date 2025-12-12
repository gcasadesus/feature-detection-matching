#pragma once

#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>

#include "lupnt/core/config.h"
#include "lupnt/core/definitions.h"
#include "lupnt/datasets/unreal_dataset.h"
#include "lupnt/slam/feature_extractor.h"
#include "lupnt/slam/feature_matcher.h"
#include "lupnt/slam/feature_set.h"

namespace lupnt {

  // 2D-2D Essential Matrix
  struct EssentialMatrixResult {
    Mat4d tgt_T_src;           // Pose
    Mat3d E;                   // Essential matrix
    std::vector<int> inliers;  // Inlier indices
  };

  class EssentialMatrixSolver {
  private:
    double threshold_ = 1.0;
    double confidence_ = 0.999;
    int max_iterations_ = 1000;

  public:
    EssentialMatrixSolver() = default;
    EssentialMatrixSolver(Config& config);
    ~EssentialMatrixSolver() = default;

    void SetThreshold(double threshold) { threshold_ = threshold; }
    void SetConfidence(double confidence) { confidence_ = confidence; }
    void SetMaxIterations(int max_iterations) { max_iterations_ = max_iterations; }

    EssentialMatrixResult Solve(const std::vector<Vec2d>& uv_1, const std::vector<Vec2d>& uv_2,
                                const Mat3d& K);
  };

  // 3D-2D PnP RANSAC
  struct PnpResult {
    Mat4d tgt_T_src;           // Pose
    std::vector<int> inliers;  // Inlier indices
    bool success;              // Whether PnP succeeded
  };

  class PnpSolver {
  private:
    double threshold_ = 1.0;
    double confidence_ = 0.999;
    int max_iterations_ = 1000;
    VecXd dist_coeffs_ = VecXd::Zero(4);

  public:
    PnpSolver() = default;
    PnpSolver(Config& config);
    ~PnpSolver() = default;

    void SetThreshold(double threshold) { threshold_ = threshold; }
    void SetConfidence(double confidence) { confidence_ = confidence; }
    void SetMaxIterations(int max_iterations) { max_iterations_ = max_iterations; }
    void SetDistCoeffs(const MatXd& dist_coeffs) { dist_coeffs_ = dist_coeffs; }

    double GetThreshold() const { return threshold_; }
    double GetConfidence() const { return confidence_; }
    int GetMaxIterations() const { return max_iterations_; }
    MatXd GetDistCoeffs() const { return dist_coeffs_; }

    PnpResult Solve(const MatX3d& xyz, const MatX2d& uv, const Mat3d& K,
                    const std::optional<Mat4d>& tgt_T_src = std::nullopt);

    // 3D-3D ICP RANSAC
    struct IcpResult {
      Mat4d tgt_T_src;           // Pose
      std::vector<int> inliers;  // Inlier indices
      bool success;              // Whether ICP converged
    };

    class IcpSolver {
    private:
      double threshold_ = 0.1;
      double confidence_ = 0.999;

    public:
      IcpSolver() = default;
      IcpSolver(Config& config);
      ~IcpSolver() = default;

      void SetThreshold(double threshold) { threshold_ = threshold; }
      void SetConfidence(double confidence) { confidence_ = confidence; }

      IcpResult Solve(const MatX3d& source_points, const MatX3d& target_points);
    };
  };

  struct TrackedPoints {
    std::vector<cv::Mat> descriptors;
    std::vector<std::vector<int>> frames;
    std::vector<Vec3d> xyz_cam;

    int Size() const { return descriptors.size(); }
  };

  class PnpVo {
  private:
    Ptr<FeatureExtractor> extractor_;
    Ptr<FeatureMatcher> matcher_;
    PnpSolver pnp_solver_;

    TrackedPoints tracked_points_;

    // Poses
    std::vector<Mat4d> poses_;

    // Previous frame
    Features feats_;        // Features in the previous frame
    ImageData img_;         // Image in the previous frame
    MatX3d xyz_;            // 3D points in the previous frame
    VecXi ids_;             // Tracked point ids
    Matches matches_;       // Matches between the current and previous frame
    PnpResult pnp_result_;  // PnP result
    int frame_idx_ = 0;     // Frame index

    bool Update(const Features& feats, const ImageData& img, const MatX3d& xyz_world);

  public:
    PnpVo(Config& config);
    ~PnpVo() = default;

    bool ProcessMono(const ImageData& img);
    bool ProcessStereo(const ImageData& img_left, const ImageData& img_right);

    // Getters
    Features GetFeatures() const { return feats_; }
    ImageData GetImage() const { return img_; }
    MatX3d GetXyz() const { return xyz_; }
    VecXi GetIds() const { return ids_; }
    Matches GetMatches() const { return matches_; }
    PnpResult GetPnpResult() const { return pnp_result_; }
    Mat4d GetPose() const { return poses_.back(); }

    TrackedPoints GetTrackedPoints() const { return tracked_points_; }
    std::vector<Mat4d> GetPoses() const { return poses_; }
  };

}  // namespace lupnt
