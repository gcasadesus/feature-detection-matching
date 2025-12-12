#include "lupnt/slam/visual_odometry.h"

#include <Eigen/src/Core/ArithmeticSequence.h>

#include "lupnt/datasets/unreal_dataset.h"
#include "lupnt/numerics/math_utils.h"
#include "lupnt/slam/feature_extractor.h"
#include "lupnt/slam/pose.h"

namespace lupnt {

  std::vector<cv::Point2f> EigenToOpencv2D(const std::vector<Vec2d>& points) {
    std::vector<cv::Point2f> cv_points;
    cv_points.reserve(points.size());
    for (const auto& point : points) {
      cv_points.emplace_back(point(0), point(1));
    }
    return cv_points;
  }

  std::vector<cv::Point3f> EigenToOpencv3D(const std::vector<Vec3d>& points) {
    std::vector<cv::Point3f> cv_points;
    cv_points.reserve(points.size());
    for (const auto& point : points) {
      cv_points.emplace_back(point(0), point(1), point(2));
    }
    return cv_points;
  }

  EssentialMatrixSolver::EssentialMatrixSolver(Config& config) {
    threshold_ = config["threshold"].as<double>(1.0);
    confidence_ = config["confidence"].as<double>(0.999);
    max_iterations_ = config["max_iterations"].as<int>(1000);

    // Log
    Logger::Debug(fmt::format("EssentialMatrixSolver initialized with threshold: {}, confidence: "
                              "{}, max_iterations: {}",
                              threshold_, confidence_, max_iterations_),
                  "EssentialMatrixSolver");
  }

  EssentialMatrixResult EssentialMatrixSolver::Solve(const std::vector<Vec2d>& points1,
                                                     const std::vector<Vec2d>& points2,
                                                     const Mat3d& K) {
    EssentialMatrixResult result;

    // Convert to OpenCV format
    std::vector<cv::Point2f> cv_points1 = EigenToOpencv2D(points1);
    std::vector<cv::Point2f> cv_points2 = EigenToOpencv2D(points2);

    // Convert camera matrix to OpenCV format using eigen2cv
    cv::Mat cv_K;
    cv::eigen2cv(K, cv_K);

    // Estimate essential matrix using RANSAC
    cv::Mat cv_E, cv_mask;
    cv_E = cv::findEssentialMat(cv_points1, cv_points2, cv_K, cv::RANSAC, confidence_, threshold_,
                                cv_mask);

    // Recover pose from essential matrix
    cv::Mat cv_R, cv_t;
    cv::recoverPose(cv_E, cv_points1, cv_points2, cv_K, cv_R, cv_t, cv_mask);

    // Convert results back to Eigen format using cv2eigen
    Mat3d R;
    Vec3d t;
    cv::cv2eigen(cv_E, result.E);
    cv::cv2eigen(cv_R, R);
    cv::cv2eigen(cv_t, t);

    result.tgt_T_src = MakeTransform(R, t);

    // Extract inliers
    result.inliers.clear();
    for (int i = 0; i < cv_mask.rows; i++) {
      if (cv_mask.at<uchar>(i) != 0) result.inliers.push_back(i);
    }

    // Log
    int n_inliers = result.inliers.size();
    double inlier_ratio = static_cast<double>(n_inliers) / points1.size();
    Logger::Debug(fmt::format("Essential matrix decomposition: {} inliers ({:.1f}%)", n_inliers,
                              inlier_ratio * 100),
                  "EssentialMatrixSolver");
    return result;
  }

  PnpSolver::PnpSolver(Config& config) {
    threshold_ = config["threshold"].as<double>(1.0);
    confidence_ = config["confidence"].as<double>(0.999);
    max_iterations_ = config["max_iterations"].as<int>(1000);
    dist_coeffs_ = config["dist_coeffs"].as<VecXd>(VecXd::Zero(4));

    // Log
    Logger::Debug(fmt::format("PnpSolver initialized with threshold: {}, confidence: {}, "
                              "max_iterations: {}",
                              threshold_, confidence_, max_iterations_),
                  "PnpSolver");
  }

  PnpResult PnpSolver::Solve(const MatX3d& object_points, const MatX2d& image_points,
                             const Mat3d& K, const std::optional<Mat4d>& tgt_T_src) {
    PnpResult result;

    // Convert to OpenCV format
    cv::Mat cv_object_points, cv_image_points;
    cv::eigen2cv(object_points, cv_object_points);
    cv::eigen2cv(image_points, cv_image_points);

    // Convert camera matrix to OpenCV format using eigen2cv
    cv::Mat cv_K;
    cv::eigen2cv(K, cv_K);

    // Convert distortion coefficients using eigen2cv
    cv::Mat cv_dist_coeffs;
    cv::eigen2cv(dist_coeffs_, cv_dist_coeffs);

    // Extrinsic guess
    Mat4d T_guess;
    if (tgt_T_src.has_value()) {
      Logger::Debug("Using extrinsic guess", "PnpSolver");
      T_guess = OCV_T_FLU * tgt_T_src.value();
    } else {
      Logger::Debug("No extrinsic guess, using identity", "PnpSolver");
      T_guess = OCV_T_FLU * Mat4d::Identity();
    }

    bool use_extrinsic_guess = true;
    cv::Mat cv_rvec, cv_tvec, cv_mask, cv_R_guess;
    Mat3d R_guess = T_guess.topLeftCorner(3, 3);
    Vec3d t_guess = T_guess.topRightCorner(3, 1);
    cv::eigen2cv(t_guess, cv_tvec);
    cv::eigen2cv(R_guess, cv_R_guess);
    cv::Rodrigues(cv_R_guess, cv_rvec);

    // Estimate pose using PnP RANSAC
    bool success = cv::solvePnPRansac(cv_object_points, cv_image_points, cv_K, cv_dist_coeffs,
                                      cv_rvec, cv_tvec, use_extrinsic_guess, max_iterations_,
                                      threshold_, confidence_, cv_mask);

    if (!success) {
      // Return identity pose if PnP fails
      Logger::Debug("PnP failed, returning identity pose", "PnpSolver");
      result.tgt_T_src = Mat4d::Identity();
      result.inliers.clear();
      result.success = false;
      return result;
    }

    // Convert rotation vector to rotation matrix
    cv::Mat cv_R;
    cv::Rodrigues(cv_rvec, cv_R);

    // Convert results back to Eigen format using cv2eigen
    Mat3d R;
    Vec3d t;
    cv::cv2eigen(cv_R, R);
    cv::cv2eigen(cv_tvec, t);

    // Extract inliers
    result.inliers.clear();
    for (int i = 0; i < cv_mask.rows; i++) {
      if (cv_mask.at<uchar>(i) != 0) result.inliers.push_back(i);
    }
    result.success = true;
    result.tgt_T_src = FLU_T_OCV * MakeTransform(R, t);

    // Log
    int n_inliers = result.inliers.size();
    double inlier_ratio = static_cast<double>(n_inliers) / object_points.rows();
    Logger::Debug(fmt::format("PnP RANSAC: {} inliers ({:.1f}%)", n_inliers, inlier_ratio * 100),
                  "PnpSolver");
    return result;
  }

  PnpSolver::IcpSolver::IcpSolver(Config& config) {
    threshold_ = config["threshold"].as<double>(0.1);
    confidence_ = config["confidence"].as<double>(0.999);

    // Log
    Logger::Debug(fmt::format("IcpSolver initialized with threshold: {}, confidence: {}",
                              threshold_, confidence_),
                  "IcpSolver");
  }

  PnpSolver::IcpResult PnpSolver::IcpSolver::Solve(const MatX3d& source_points,
                                                   const MatX3d& target_points) {
    IcpResult result;
    result.success = false;
    if (source_points.rows() == 0 || target_points.rows() == 0) {
      result.tgt_T_src = Mat4d::Identity();
      result.inliers.clear();
      return result;
    }

    // Convert to OpenCV format
    cv::Mat cv_source_points, cv_target_points;
    cv::eigen2cv(source_points, cv_source_points);
    cv::eigen2cv(target_points, cv_target_points);

    // Use OpenCV's estimateAffine3D for robust 3D transformation estimation
    cv::Mat cv_affine_transform, cv_mask;
    bool success = cv::estimateAffine3D(cv_source_points, cv_target_points, cv_affine_transform,
                                        cv_mask, threshold_, confidence_);

    if (!success || cv_affine_transform.empty()) {
      // Return identity pose if estimation fails
      result.tgt_T_src = Mat4d::Identity();
      result.inliers.clear();
      return result;
    }

    // Extract rotation and translation from affine transformation matrix
    // The affine transform is a 3x4 matrix [R|t] where R is 3x3 rotation and t is 3x1 translation
    Mat3d R;
    Vec3d t;

    // Convert the affine transformation matrix to Eigen
    cv::cv2eigen(cv_affine_transform.colRange(0, 3), R);  // First 3 columns (rotation)
    cv::cv2eigen(cv_affine_transform.col(3), t);          // Last column (translation)

    // Ensure proper rotation matrix (orthogonal and det(R) = 1)
    Eigen::JacobiSVD<Mat3d> svd(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Mat3d U = svd.matrixU();
    Mat3d V = svd.matrixV();
    R = U * V.transpose();

    // Ensure proper rotation matrix (det(R) = 1)
    if (R.determinant() < 0) {
      V.col(2) *= -1;
      R = U * V.transpose();
    }

    // Extract inliers
    result.inliers.clear();
    for (int i = 0; i < cv_mask.rows; i++) {
      if (cv_mask.at<uchar>(i) != 0) result.inliers.push_back(i);
    }
    result.tgt_T_src = MakeTransform(R, t);
    result.success = true;  // estimateAffine3D either succeeds or fails, no iteration

    // Log
    int n_inliers = result.inliers.size();
    double inlier_ratio = static_cast<double>(n_inliers) / source_points.rows();
    Logger::Debug(fmt::format("ICP: {} inliers ({:.1f}%)", n_inliers, inlier_ratio * 100),
                  "IcpSolver");
    return result;
  }

  PnpVo::PnpVo(Config& config) {
    auto extractor_config = config["feature_extractor"];
    auto matcher_config = config["feature_matcher"];
    auto pnp_solver_config = config["pnp_solver"];

    extractor_ = FeatureExtractor::FromConfig(extractor_config);
    matcher_ = FeatureMatcher::FromConfig(matcher_config);
    pnp_solver_ = PnpSolver(pnp_solver_config);

    if (config["initial_pose"].IsDefined()) {
      poses_.push_back(config["initial_pose"].as<MatXd>());
    } else {
      poses_.push_back(Mat4d::Identity());
    }

    // Log
    Logger::Debug("PnpVo initialized", "PnpVo");
  }

  MatX3d ApplyTransform(const Mat4d& T, const MatX3d& xyz) {
    Mat3d R = T.block<3, 3>(0, 0);
    Vec3d t = T.block<3, 1>(0, 3);
    MatX3d xyz1 = xyz * R.transpose() + t.transpose().replicate(xyz.rows(), 1);
    return xyz1;
  }

  MatX3d PixelsToPoints(const MatX2d& uv, const VecXd& depth, const CameraIntrinsics intrinsics) {
    MatX3d xyz_ocv = MatX3d::Zero(uv.rows(), 3);
    xyz_ocv.col(0) = (uv.col(0).array() - intrinsics.cx) * depth.array() / intrinsics.fx;
    xyz_ocv.col(1) = (uv.col(1).array() - intrinsics.cy) * depth.array() / intrinsics.fy;
    xyz_ocv.col(2) = depth.array();
    MatX3d xyz_cam = ApplyTransform(FLU_T_OCV, xyz_ocv);
    return xyz_cam;
  }

  MatX2d PointsToPixels(const MatX3d& xyz_cam, const CameraIntrinsics intrinsics) {
    MatX3d xyz_ocv = ApplyTransform(OCV_T_FLU, xyz_cam);
    MatX2d uv = MatX2d::Zero(xyz_ocv.rows(), 2);
    uv.col(0) = (xyz_ocv.col(0).array() * intrinsics.fx / xyz_ocv.col(2).array()) + intrinsics.cx;
    uv.col(1) = (xyz_ocv.col(1).array() * intrinsics.fy / xyz_ocv.col(2).array()) + intrinsics.cy;
    return uv;
  }

  Mat3 MakeCameraMatrix(const CameraIntrinsics& intrinsics) {
    Real fx = intrinsics.fx, fy = intrinsics.fy, cx = intrinsics.cx, cy = intrinsics.cy;
    Mat3 K{{fx, 0, cx}, {0, fy, cy}, {0, 0, 1}};
    return K;
  }

  bool PnpVo::Update(const Features& feats, const ImageData& img, const MatX3d& xyz_cam) {
    matches_ = Matches{MatX2i::Zero(0, 2), VecXd::Zero(0)};
    if (tracked_points_.Size() == 0) {
      Logger::Debug("Initializing first frame", "PnpVo");
    } else {
      // Match
      matches_ = matcher_->Match(feats_, feats);
      Logger::Debug(fmt::format("Found {} matches", matches_.Size()), "PnpVo");

      // Check
      if (matches_.Size() < 4) {
        Logger::Warn("Not enough matches for PnP, skipping update", "PnpVo");
        return false;
      }

      // PnP
      MatX2d uv_curr = feats.uv(matches_.indexes.col(1), Eigen::all);
      MatX2d uv_prev = feats_.uv(matches_.indexes.col(0), Eigen::all);
      MatX3d xyz_prev_matched = xyz_(matches_.indexes.col(0), Eigen::all);
      Mat3d K = MakeCameraMatrix(img.intrinsics);
      pnp_result_ = pnp_solver_.Solve(xyz_prev_matched, uv_curr, K);

      // Check
      if (!pnp_result_.success) {
        Logger::Warn("PnP failed, keeping previous pose", "PnpVo");
        return false;
      }

      // Pose
      Mat4d bTc_prev = img_.body_T_cam;
      Mat4d cTb_curr = InvertTransform(img.body_T_cam);
      Mat4d c_prevTc_curr = InvertTransform(pnp_result_.tgt_T_src);
      Mat4d wTb_prev = poses_.back();
      Mat4d wTb_curr = wTb_prev * bTc_prev * c_prevTc_curr * cTb_curr;
      poses_.push_back(wTb_curr);
    }

    // Update tracked points
    VecXi feat_to_match = VecXi::Ones(feats.Size()).array() * -1;
    feat_to_match(matches_.indexes.col(1)) = Arange<int>(0, matches_.Size(), 1);

    VecXi ids(feats.Size());
    int n_new_points = 0;
    for (int i = 0; i < feats.Size(); i++) {
      int j = feat_to_match(i);
      if (j == -1) {  // New point
        ids[i] = tracked_points_.Size();
        tracked_points_.descriptors.push_back(feats.descriptors.row(i).clone());
        tracked_points_.frames.push_back({frame_idx_});
        tracked_points_.xyz_cam.push_back(xyz_cam.row(i));
        n_new_points++;
      } else {  // Existing point
        int id = ids_(matches_.indexes.row(j)(0));
        tracked_points_.frames[id].push_back(frame_idx_);
        ids[i] = id;
      }
    }
    Logger::Debug(fmt::format("Tracking {} points ({} new)", tracked_points_.Size(), n_new_points),
                  "PnpVo");

    // Update
    feats_ = feats;
    img_ = img;
    xyz_ = xyz_cam;
    ids_ = ids;
    frame_idx_++;
    return true;
  }

  bool PnpVo::ProcessMono(const ImageData& img) {
    // Features
    auto feats = extractor_->Extract(img.rgb);
    Logger::Debug(fmt::format("Extracted {} features", feats.Size()), "PnpVo");

    // Depth
    MatXd depth_mat;
    cv::cv2eigen(img.depth, depth_mat);
    VecXd depth = depth_mat(feats.uv.col(1), feats.uv.col(0));
    MatX3d xyz_cam = PixelsToPoints(feats.uv, depth, img.intrinsics);

    return Update(feats, img, xyz_cam);
  }

  bool PnpVo::ProcessStereo(const ImageData& img_left, const ImageData& img_right) {
    // Features
    auto feats_left = extractor_->Extract(img_left.rgb);
    auto feats_right = extractor_->Extract(img_right.rgb);
    auto matches_left_right = matcher_->Match(feats_left, feats_right);

    // Depth
    Vec3d t_left = img_left.body_T_cam.block<3, 1>(0, 3);
    Vec3d t_right = img_right.body_T_cam.block<3, 1>(0, 3);
    double baseline = (t_left - t_right).norm();
    Vec3d disp = feats_left.uv.col(0).array() - feats_right.uv.col(0).array();
    VecXd depth = baseline / (disp.array() + 1e-8);
    MatX3d xyz_cam = PixelsToPoints(feats_left.uv, depth, img_left.intrinsics);

    return Update(feats_left, img_left, xyz_cam);
  }

}  // namespace lupnt
