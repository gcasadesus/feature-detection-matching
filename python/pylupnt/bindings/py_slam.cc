#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include <unordered_map>

#include "lupnt/core/asset_factory.h"
#include "lupnt/slam/feature_extractor.h"
#include "lupnt/slam/feature_matcher.h"
#include "lupnt/slam/feature_set.h"
#include "lupnt/slam/visual_odometry.h"
#include "py_pybind11.h"  // NOLINT(misc-include-cleaner)
#include "py_yaml.h"      // NOLINT(misc-include-cleaner)

namespace py = pybind11;
using namespace lupnt;

// Trampoline class for Python inheritance
class PyFeatureExtractor : public FeatureExtractor {
public:
  using FeatureExtractor::FeatureExtractor;

  Features Extract(const cv::Mat& image) override {
    PYBIND11_OVERRIDE_PURE_NAME(Features, FeatureExtractor, "extract", Extract, image);
  }

  std::string GetName() const override {
    PYBIND11_OVERRIDE_PURE_NAME(std::string, FeatureExtractor, "get_name", GetName);
  }
};

class PyFeatureMatcher : public FeatureMatcher {
public:
  using FeatureMatcher::FeatureMatcher;

  Matches Match(const Features& features1, const Features& features2) override {
    PYBIND11_OVERRIDE_PURE_NAME(Matches, FeatureMatcher, "match", Match, features1, features2);
  }

  std::string GetName() const override {
    PYBIND11_OVERRIDE_PURE_NAME(std::string, FeatureMatcher, "get_name", GetName);
  }
};

void InitSlam(py::module& m) {
  // Features
  py::class_<Features>(m, "Features")
      .def(py::init<>())
      .def(py::init<const std::vector<cv::KeyPoint>&, const cv::Mat&>())
      .def_readwrite("descriptors", &Features::descriptors)
      .def_readwrite("scores", &Features::scores)
      .def_readwrite("scales", &Features::scales)
      .def_readwrite("orientations", &Features::orientations)
      .def_readwrite("image", &Features::image)
      .def_readwrite("uv", &Features::uv)
      .def("size", &Features::Size)
      .def("empty", &Features::Empty)
      .def("filter_best", &Features::FilterBest, py::arg("n"))
      .def("filter", &Features::Filter, py::arg("indices"))
      .def("__len__", &Features::Size)
      .def("__repr__",
           [](const Features& fs) {
             return "<Features with " + std::to_string(fs.Size()) + " features>";
           })
      .def(py::pickle(
          [](const Features& fs) {  // __getstate__
            return py::make_tuple(fs.descriptors, fs.uv, fs.scores, fs.scales, fs.orientations,
                                  fs.image);
          },
          [](py::tuple t) {  // __setstate__
            if (t.size() != 6) {
              LUPNT_CHECK(false, "Invalid state for Features!", "Features");
            }
            Features fs;
            fs.descriptors = t[0].cast<cv::Mat>();
            fs.uv = t[1].cast<MatX2d>();
            fs.scores = t[2].cast<VecXd>();
            fs.scales = t[3].cast<VecXd>();
            fs.orientations = t[4].cast<VecXd>();
            fs.image = t[5].cast<cv::Mat>();
            return fs;
          }));

  // Matches
  py::class_<Matches>(m, "Matches")
      .def(py::init<>())
      .def(py::init<const MatX2i&, const VecXd&>(), py::arg("indexes"), py::arg("distances"))
      .def_readwrite("indexes", &Matches::indexes)
      .def_readwrite("distances", &Matches::distances)
      .def("__len__", &Matches::Size)
      .def("__repr__",
           [](const Matches& mr) {
             return "<Matches with " + std::to_string(mr.Size()) + " matches>";
           })
      .def(py::pickle(
          [](const Matches& mr) {  // __getstate__
            return py::make_tuple(mr.indexes, mr.distances);
          },
          [](py::tuple t) {  // __setstate__
            if (t.size() != 2) {
              LUPNT_CHECK(false, "Invalid state for Matches!", "Matches");
            }
            Matches mr;
            mr.indexes = t[0].cast<MatX2i>();
            mr.distances = t[1].cast<VecXd>();
            return mr;
          }));

  // OpenCV KeyPoint
  py::class_<cv::KeyPoint>(m, "KeyPoint")
      .def(py::init<>())
      .def(py::init<float, float, float>(), py::arg("x"), py::arg("y"), py::arg("size"))
      .def(py::init<float, float, float, float, float, int, int>(), py::arg("x"), py::arg("y"),
           py::arg("size"), py::arg("angle") = -1, py::arg("response") = 0, py::arg("octave") = 0,
           py::arg("class_id") = -1)
      .def_readwrite("pt", &cv::KeyPoint::pt)
      .def_property(
          "x", [](const cv::KeyPoint& kp) { return kp.pt.x; },
          [](cv::KeyPoint& kp, float x) { kp.pt.x = x; })
      .def_property(
          "y", [](const cv::KeyPoint& kp) { return kp.pt.y; },
          [](cv::KeyPoint& kp, float y) { kp.pt.y = y; })
      .def_readwrite("size", &cv::KeyPoint::size)
      .def_readwrite("angle", &cv::KeyPoint::angle)
      .def_readwrite("response", &cv::KeyPoint::response)
      .def_readwrite("octave", &cv::KeyPoint::octave)
      .def_readwrite("class_id", &cv::KeyPoint::class_id)
      .def("__repr__", [](const cv::KeyPoint& kp) {
        return "<KeyPoint x=" + std::to_string(kp.pt.x) + " y=" + std::to_string(kp.pt.y)
               + " size=" + std::to_string(kp.size) + ">";
      });

  // OpenCV Point2f (needed for KeyPoint.pt)
  py::class_<cv::Point2f>(m, "Point2f")
      .def(py::init<>())
      .def(py::init<float, float>(), py::arg("x"), py::arg("y"))
      .def_readwrite("x", &cv::Point2f::x)
      .def_readwrite("y", &cv::Point2f::y)
      .def("__repr__", [](const cv::Point2f& p) {
        return "<Point2f x=" + std::to_string(p.x) + " y=" + std::to_string(p.y) + ">";
      });

  // OpenCV DMatch
  py::class_<cv::DMatch>(m, "DMatch")
      .def(py::init<>())
      .def(py::init<int, int, float>(), py::arg("query_idx"), py::arg("train_idx"),
           py::arg("distance"))
      .def(py::init<int, int, int, float>(), py::arg("query_idx"), py::arg("train_idx"),
           py::arg("img_idx"), py::arg("distance"))
      .def_readwrite("queryIdx", &cv::DMatch::queryIdx)
      .def_readwrite("trainIdx", &cv::DMatch::trainIdx)
      .def_readwrite("imgIdx", &cv::DMatch::imgIdx)
      .def_readwrite("distance", &cv::DMatch::distance)
      .def("__repr__", [](const cv::DMatch& m) {
        return "<DMatch queryIdx=" + std::to_string(m.queryIdx) + " trainIdx="
               + std::to_string(m.trainIdx) + " distance=" + std::to_string(m.distance) + ">";
      });

  // Feature Extractor
  auto feature_extractor_class
      = py::class_<FeatureExtractor, PyFeatureExtractor, std::shared_ptr<FeatureExtractor>>(
            m, "FeatureExtractor")
            .def(py::init<>())
            .def("extract", &FeatureExtractor::Extract, py::arg("image"))
            .def("get_name", &FeatureExtractor::GetName, "Get the name of the extractor")
            .def_static("from_config", &FeatureExtractor::FromConfig, py::arg("config"))
            .def_static(
                "register",
                [](const std::string& class_name, py::object py_class) {
                  auto py_class_ptr = std::make_shared<py::object>(py_class);
                  AssetFactory<FeatureExtractor, Config&>::Register(
                      class_name,
                      [py_class_ptr](Config& config) -> std::shared_ptr<FeatureExtractor> {
                        py::gil_scoped_acquire acquire;
                        py::object instance = (*py_class_ptr)(config);
                        // Keep Python object alive by using pybind11's shared_ptr holder
                        auto ptr = instance.cast<std::shared_ptr<FeatureExtractor>>();
                        // Store Python object to keep it alive
                        static std::unordered_map<void*, py::object> py_objects;
                        py_objects[ptr.get()] = instance;
                        return ptr;
                      });
                },
                py::arg("class_name"), py::arg("py_class"),
                "Register a Python FeatureExtractor class");

  // OpenCV Feature Extractor base class
  py::class_<OpencvFeatureExtractor, FeatureExtractor, std::shared_ptr<OpencvFeatureExtractor>>(
      m, "OpencvFeatureExtractor")
      .def("get_detector", &OpencvFeatureExtractor::GetDetector);

  // SIFT
  py::class_<Sift, OpencvFeatureExtractor, std::shared_ptr<Sift>>(m, "Sift").def(
      py::init<Config&>(), py::arg("config"));

  // ORB
  py::class_<Orb, OpencvFeatureExtractor, std::shared_ptr<Orb>>(m, "Orb").def(py::init<Config&>(),
                                                                              py::arg("config"));

  // AKAZE
  py::class_<Akaze, OpencvFeatureExtractor, std::shared_ptr<Akaze>>(m, "Akaze")
      .def(py::init<Config&>(), py::arg("config"));

  // BRISK
  py::class_<Brisk, OpencvFeatureExtractor, std::shared_ptr<Brisk>>(m, "Brisk")
      .def(py::init<Config&>(), py::arg("config"));

  // Feature Matcher
  auto feature_matcher_class
      = py::class_<FeatureMatcher, PyFeatureMatcher, std::shared_ptr<FeatureMatcher>>(
            m, "FeatureMatcher")
            .def(py::init<>())
            .def("match", &FeatureMatcher::Match, py::arg("features1"), py::arg("features2"))
            .def("get_name", &FeatureMatcher::GetName)
            .def_static("from_config", &FeatureMatcher::FromConfig, py::arg("config"))
            .def_static(
                "register",
                [](const std::string& class_name, py::object py_class) {
                  auto py_class_ptr = std::make_shared<py::object>(py_class);
                  AssetFactory<FeatureMatcher, Config&>::Register(
                      class_name,
                      [py_class_ptr](Config& config) -> std::shared_ptr<FeatureMatcher> {
                        py::gil_scoped_acquire acquire;
                        py::object instance = (*py_class_ptr)(config);
                        // Keep Python object alive by using pybind11's shared_ptr holder
                        auto ptr = instance.cast<std::shared_ptr<FeatureMatcher>>();
                        // Store Python object to keep it alive
                        static std::unordered_map<void*, py::object> py_objects;
                        py_objects[ptr.get()] = instance;
                        return ptr;
                      });
                },
                py::arg("class_name"), py::arg("py_class"),
                "Register a Python FeatureMatcher class");

  // OpenCV Feature Matcher base class
  py::class_<OpencvFeatureMatcher, FeatureMatcher, std::shared_ptr<OpencvFeatureMatcher>>(
      m, "OpencvFeatureMatcher");

  // FLANN-based Matcher
  py::class_<FlannMatcher, OpencvFeatureMatcher, std::shared_ptr<FlannMatcher>>(m, "FlannMatcher")
      .def(py::init<Config&>(), py::arg("config"));

  // Brute Force Matcher
  py::class_<BruteForceMatcher, OpencvFeatureMatcher, std::shared_ptr<BruteForceMatcher>>(
      m, "BruteForceMatcher")
      .def(py::init<Config&>(), py::arg("config"));

  // Essential Matrix
  py::class_<EssentialMatrixResult>(m, "EssentialMatrixResult")
      .def_readwrite("tgt_T_src", &EssentialMatrixResult::tgt_T_src)
      .def_readwrite("E", &EssentialMatrixResult::E)
      .def_readwrite("inliers", &EssentialMatrixResult::inliers);
  py::class_<EssentialMatrixSolver>(m, "EssentialMatrixSolver")
      .def(py::init<>())
      .def(py::init<Config&>(), py::arg("config"))
      .def("set_threshold", &EssentialMatrixSolver::SetThreshold, py::arg("threshold"))
      .def("set_confidence", &EssentialMatrixSolver::SetConfidence, py::arg("confidence"))
      .def("set_max_iterations", &EssentialMatrixSolver::SetMaxIterations,
           py::arg("max_iterations"))
      .def("solve", &EssentialMatrixSolver::Solve, py::arg("points1"), py::arg("points2"),
           py::arg("K"));

  // PnP
  py::class_<PnpResult>(m, "PnpResult")
      .def_readwrite("tgt_T_src", &PnpResult::tgt_T_src)
      .def_readwrite("inliers", &PnpResult::inliers)
      .def_readwrite("success", &PnpResult::success);
  py::class_<PnpSolver>(m, "PnpSolver")
      .def(py::init<>())
      .def(py::init<Config&>(), py::arg("config"))
      .def("solve", &PnpSolver::Solve, py::arg("object_points"), py::arg("image_points"),
           py::arg("K"), py::arg("tgt_T_src") = std::nullopt);

  // ICP
  py::class_<PnpSolver::IcpResult>(m, "IcpResult")
      .def_readwrite("tgt_T_src", &PnpSolver::IcpResult::tgt_T_src)
      .def_readwrite("inliers", &PnpSolver::IcpResult::inliers)
      .def_readwrite("success", &PnpSolver::IcpResult::success);
  py::class_<PnpSolver::IcpSolver>(m, "IcpSolver")
      .def(py::init<>())
      .def(py::init<Config&>(), py::arg("config"))
      .def("solve", &PnpSolver::IcpSolver::Solve, py::arg("source_points"),
           py::arg("target_points"));

  // ImageData - can be constructed from dict or created and populated via attributes
  py::class_<ImageData>(m, "ImageData")
      .def(py::init<>())
      .def(py::init<const Config&>(), py::arg("config"))
      .def_readwrite("rgb", &ImageData::rgb)
      .def_readwrite("depth", &ImageData::depth)
      .def_readwrite("label", &ImageData::label)
      .def_readwrite("intrinsics", &ImageData::intrinsics)
      .def_readwrite("body_T_cam", &ImageData::body_T_cam)
      .def_readwrite("world_T_cam", &ImageData::world_T_cam)
      .def("__getitem__",
           [](const ImageData& self, const std::string& key) -> py::object {
             if (key == "rgb") return py::cast(self.rgb);
             if (key == "depth") return py::cast(self.depth);
             if (key == "label") return py::cast(self.label);
             if (key == "intrinsics") return py::cast(self.intrinsics);
             if (key == "body_T_cam") return py::cast(self.body_T_cam);
             if (key == "world_T_cam") return py::cast(self.world_T_cam);
             throw py::key_error("Key '" + key + "' not found in ImageData");
           })
      .def("__setitem__",
           [](ImageData& self, const std::string& key, py::handle value) {
             if (key == "rgb") {
               self.rgb = value.cast<cv::Mat>();
             } else if (key == "depth") {
               self.depth = value.cast<cv::Mat>();
             } else if (key == "label") {
               self.label = value.cast<cv::Mat>();
             } else if (key == "intrinsics") {
               self.intrinsics = value.cast<CameraIntrinsics>();
             } else if (key == "body_T_cam") {
               self.body_T_cam = value.cast<Mat4d>();
             } else if (key == "world_T_cam") {
               self.world_T_cam = value.cast<Mat4d>();
             } else {
               throw py::key_error("Key '" + key + "' not found in ImageData");
             }
           })
      .def("__contains__",
           [](const ImageData&, const std::string& key) {
             return key == "rgb" || key == "depth" || key == "label" || key == "intrinsics"
                    || key == "body_T_cam" || key == "world_T_cam";
           })
      .def("keys",
           []() {
             return py::make_tuple("rgb", "depth", "label", "intrinsics", "body_T_cam",
                                   "world_T_cam");
           })
      .def("__repr__",
           [](const ImageData& self) {
             int rows = self.rgb.rows, cols = self.rgb.cols;
             bool has_depth = !self.depth.empty();
             bool has_label = !self.label.empty();
             return "<ImageData rows=" + std::to_string(rows) + " cols=" + std::to_string(cols)
                    + " depth=" + (has_depth ? "True" : "False")
                    + " label=" + (has_label ? "True" : "False") + ">";
           })
      .def("get", [](const ImageData& self, const std::string& key) -> py::object {
        if (key == "rgb") return py::cast(self.rgb);
        if (key == "depth") return py::cast(self.depth);
        if (key == "label") return py::cast(self.label);
        if (key == "intrinsics") return py::cast(self.intrinsics);
        if (key == "body_T_cam") return py::cast(self.body_T_cam);
        if (key == "world_T_cam") return py::cast(self.world_T_cam);
        return py::none();
      });
  // Intrinsics
  py::class_<CameraIntrinsics>(m, "CameraIntrinsics")
      .def(py::init<>())
      .def(py::init<const Config&>(), py::arg("config"))
      .def_readwrite("fx", &CameraIntrinsics::fx)
      .def_readwrite("fy", &CameraIntrinsics::fy)
      .def_readwrite("cx", &CameraIntrinsics::cx)
      .def_readwrite("cy", &CameraIntrinsics::cy)
      .def_readwrite("W", &CameraIntrinsics::W)
      .def_readwrite("H", &CameraIntrinsics::H)
      .def("__getitem__",
           [](const CameraIntrinsics& self, const std::string& key) -> py::object {
             if (key == "fx") return py::cast(self.fx);
             if (key == "fy") return py::cast(self.fy);
             if (key == "cx") return py::cast(self.cx);
             if (key == "cy") return py::cast(self.cy);
             if (key == "W") return py::cast(self.W);
             if (key == "H") return py::cast(self.H);
             throw py::key_error("Key '" + key + "' not found in CameraIntrinsics");
           })
      .def("__setitem__",
           [](CameraIntrinsics& self, const std::string& key, py::handle value) {
             if (key == "fx") {
               self.fx = value.cast<double>();
             } else if (key == "fy") {
               self.fy = value.cast<double>();
             } else if (key == "cx") {
               self.cx = value.cast<double>();
             } else if (key == "cy") {
               self.cy = value.cast<double>();
             } else if (key == "W") {
               self.W = value.cast<int>();
             } else if (key == "H") {
               self.H = value.cast<int>();
             } else {
               throw py::key_error("Key '" + key + "' not found in CameraIntrinsics");
             }
           })
      .def("__contains__",
           [](const CameraIntrinsics&, const std::string& key) {
             return key == "fx" || key == "fy" || key == "cx" || key == "cy" || key == "W"
                    || key == "H";
           })
      .def("keys", []() { return py::make_tuple("fx", "fy", "cx", "cy", "W", "H"); })
      .def("__repr__", [](const CameraIntrinsics& self) {
        return "<CameraIntrinsics fx=" + std::to_string(self.fx) + " fy=" + std::to_string(self.fy)
               + " cx=" + std::to_string(self.cx) + " cy=" + std::to_string(self.cy)
               + " W=" + std::to_string(self.W) + " H=" + std::to_string(self.H) + ">";
      });

  // PnpVo
  py::class_<PnpVo>(m, "PnpVo")
      .def(py::init<Config&>(), py::arg("config"))
      .def("process_mono", &PnpVo::ProcessMono, py::arg("img"))
      .def("process_stereo", &PnpVo::ProcessStereo, py::arg("img_left"), py::arg("img_right"))
      .def("get_poses", &PnpVo::GetPoses)
      .def("get_pose", &PnpVo::GetPose)
      .def("get_features", &PnpVo::GetFeatures)
      .def("get_matches", &PnpVo::GetMatches)
      .def("get_pnp_result", &PnpVo::GetPnpResult)
      .def("get_tracked_points", &PnpVo::GetTrackedPoints);

  // TrackedPoints
  py::class_<TrackedPoints>(m, "TrackedPoints")
      .def_readwrite("descriptors", &TrackedPoints::descriptors)
      .def_readwrite("frames", &TrackedPoints::frames)
      .def_readwrite("xyz_cam", &TrackedPoints::xyz_cam)
      .def("size", &TrackedPoints::Size);
}
