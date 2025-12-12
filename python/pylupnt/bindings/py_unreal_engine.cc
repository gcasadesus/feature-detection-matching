#include <lupnt/core/error.h>
#include <lupnt/interfaces/unreal_engine.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>

#include <mutex>
#include <opencv2/core.hpp>
#include <opencv2/core/core.hpp>
#include <unordered_map>

#include "py_pybind11.h"
#include "py_yaml.h"

namespace py = pybind11;
using namespace lupnt;

void InitUnrealEngine(py::module& m) {
  // Coordinate conversion utilities
  m.def("xyz_to_xyz_unreal", &XyzToXyzUnreal, py::arg("xyz"));
  m.def("xyz_unreal_to_xyz", &XyzUnrealToXyz, py::arg("xyz_unreal"));
  m.def("rpy_to_rpy_unreal", &RpyToRpyUnreal, py::arg("rpy"));
  m.def("rpy_unreal_to_rpy", &RpyUnrealToRpy, py::arg("rpy_unreal"));
  m.def("rot_to_rpy_unreal", &RotToRpyUnreal, py::arg("rot"));
  m.def("rpy_unreal_to_rot", &RpyUnrealToRot, py::arg("rpy_unreal"));

  static std::unordered_map<UnrealEngine*, int> context_depth;
  static std::mutex context_mutex;

  auto unreal_engine
      = py::class_<UnrealEngine>(m, "UnrealEngine")
            .def(py::init<const std::string&, int>(), py::arg("host") = UnrealEngine::HOST,
                 py::arg("port") = UnrealEngine::TCP_PORT)
            .def("connect", &UnrealEngine::Connect)
            .def("disconnect", &UnrealEngine::Disconnect)
            .def("__enter__",
                 [](UnrealEngine& self) -> UnrealEngine& {
                   bool should_connect = false;
                   {
                     std::lock_guard<std::mutex> lock(context_mutex);
                     int& depth = context_depth[&self];
                     if (depth == 0) should_connect = true;
                     ++depth;
                   }
                   if (should_connect) {
                     self.Connect();
                     // Initialize shared memory with 2GB default (enough for ~70 3K images)
                     self.InitSharedMemory("/lupnt_render_shm", 2000);
                   }
                   return self;
                 })
            .def("__exit__",
                 [](UnrealEngine& self, py::object /*exc_type*/, py::object /*exc_value*/,
                    py::object /*traceback*/) {
                   bool should_disconnect = false;
                   {
                     std::lock_guard<std::mutex> lock(context_mutex);
                     auto it = context_depth.find(&self);
                     if (it != context_depth.end()) {
                       int& depth = it->second;
                       if (--depth <= 0) {
                         context_depth.erase(it);
                         should_disconnect = true;
                       }
                     } else {
                       should_disconnect = true;
                     }
                   }
                   if (should_disconnect) {
                     self.Disconnect();
                   }
                 })
            .def("spawn",
                 py::overload_cast<const std::string&, const std::string&, const Vec3&, const Vec3&,
                                   const Config&>(&UnrealEngine::Spawn),
                 py::arg("agent_id"), py::arg("type"), py::arg("xyz"), py::arg("rpy"),
                 py::arg("config") = Config())
            .def("spawn",
                 py::overload_cast<const std::string&, const std::string&, const Vec3&, const Mat3&,
                                   const Config&>(&UnrealEngine::Spawn),
                 py::arg("agent_id"), py::arg("type"), py::arg("xyz"), py::arg("rot"),
                 py::arg("config") = Config())
            .def("set_actor_transform",
                 py::overload_cast<const std::string&, const Vec3&, const Vec3&>(
                     &UnrealEngine::SetActorTransform),
                 py::arg("agent_id"), py::arg("xyz"), py::arg("rpy"))
            .def("set_actor_transform",
                 py::overload_cast<const std::string&, const Vec3&, const Mat3&>(
                     &UnrealEngine::SetActorTransform),
                 py::arg("agent_id"), py::arg("xyz"), py::arg("rot"))
            .def("set_control", &UnrealEngine::SetControl, py::arg("agent_id"),
                 py::arg("throttle_value"), py::arg("steering_value"))
            .def("set_pitch_yaw", &UnrealEngine::SetPitchYaw, py::arg("agent_id"),
                 py::arg("pitch_value"), py::arg("yaw_value"))
            .def("set_sun", &UnrealEngine::SetSun, py::arg("azimuth"), py::arg("elevation"))
            .def("render", &UnrealEngine::Render, py::arg("agent_id"), py::arg("camera"),
                 py::arg("type"), py::arg("width") = std::nullopt, py::arg("height") = std::nullopt,
                 py::arg("fov") = std::nullopt, py::arg("ss_factor") = std::nullopt)
            .def(
                "batch_render",
                [](UnrealEngine& self, const py::list& requests_list) {
                  std::vector<UnrealEngine::RenderRequest> requests;
                  for (auto item : requests_list) {
                    auto req_dict = item.cast<py::dict>();
                    UnrealEngine::RenderRequest req;
                    req.agent_id = req_dict["agent_id"].cast<std::string>();
                    req.camera = req_dict["camera"].cast<std::string>();
                    req.render_type = req_dict["render_type"].cast<std::string>();
                    if (req_dict.contains("width")) req.width = req_dict["width"].cast<int32_t>();
                    if (req_dict.contains("height"))
                      req.height = req_dict["height"].cast<int32_t>();
                    if (req_dict.contains("fov")) req.fov = req_dict["fov"].cast<float>();
                    if (req_dict.contains("ss_factor"))
                      req.ss_factor = req_dict["ss_factor"].cast<float>();
                    if (req_dict.contains("grayscale"))
                      req.grayscale = req_dict["grayscale"].cast<bool>();
                    requests.push_back(req);
                  }
                  return self.BatchRender(requests);
                },
                py::arg("requests"))
            .def("init_shared_memory", &UnrealEngine::InitSharedMemory,
                 py::arg("name") = "/lupnt_render_shm", py::arg("size_mb") = 2000)
            .def("shutdown_shared_memory", &UnrealEngine::ShutdownSharedMemory)
            .def("clear_render_target_pool", &UnrealEngine::ClearRenderTargetPool)
            .def("reset_camera_properties", &UnrealEngine::ResetCameraProperties,
                 py::arg("agent_id"), py::arg("camera_name"))
            .def("set_target_view", &UnrealEngine::SetTargetView, py::arg("agent_id"))
            // Component Management
            .def("add_to_agent", &UnrealEngine::AddToAgent, py::arg("agent_id"), py::arg("type"),
                 py::arg("name"), py::arg("params") = Config())
            .def("remove_from_agent", &UnrealEngine::RemoveFromAgent, py::arg("agent_id"),
                 py::arg("type"), py::arg("name"))
            .def("set_component_transform",
                 py::overload_cast<const std::string&, const std::string&, const std::string&,
                                   const Vec3&, const Vec3&>(&UnrealEngine::SetComponentTransform),
                 py::arg("agent_id"), py::arg("type"), py::arg("name"), py::arg("xyz"),
                 py::arg("rpy"))
            .def("set_component_transform",
                 py::overload_cast<const std::string&, const std::string&, const std::string&,
                                   const Vec3&, const Mat3&>(&UnrealEngine::SetComponentTransform),
                 py::arg("agent_id"), py::arg("type"), py::arg("name"), py::arg("xyz"),
                 py::arg("rot"))
            .def("set_component_properties", &UnrealEngine::SetComponentProperties,
                 py::arg("agent_id"), py::arg("type"), py::arg("name"), py::arg("properties"))
            .def("pause", &UnrealEngine::Pause)
            .def("play", &UnrealEngine::Play)
            .def("step", &UnrealEngine::Step, py::arg("duration"))
            .def("set_time_dilation", &UnrealEngine::SetTimeDilation, py::arg("time_dilation"))
            .def("set_timestep", &UnrealEngine::SetTimestep, py::arg("timestep"))
            .def("get_state", &UnrealEngine::GetState, py::arg("agent_id"))
            .def("get_agent_components", &UnrealEngine::GetAgentComponents, py::arg("agent_id"))
            .def("remove", &UnrealEngine::Remove, py::arg("agent_id"))
            .def("remove_all", &UnrealEngine::RemoveAll)
            .def("send_request", &UnrealEngine::SendRequest, py::arg("msg_json"))
            .def("receive_render", &UnrealEngine::ReceiveRender, py::arg("render_type"));

  unreal_engine.attr("REFERENCE_HEIGHT") = UnrealEngine::REFERENCE_HEIGHT;
  unreal_engine.attr("UE_REFERENCE_MOON_PA") = UnrealEngine::UE_REFERENCE_MOON_PA;
  unreal_engine.attr("MAX_DEPTH") = UnrealEngine::MAX_DEPTH;
  unreal_engine.attr("Label") = py::enum_<UnrealEngine::Label>(unreal_engine, "Label")
                                    .value("SKY", UnrealEngine::Label::SKY)
                                    .value("REGOLITH", UnrealEngine::Label::REGOLITH)
                                    .value("ROVER", UnrealEngine::Label::ROVER)
                                    .value("ROCK", UnrealEngine::Label::ROCK)
                                    .value("LANDER", UnrealEngine::Label::LANDER)
                                    .value("SUN", UnrealEngine::Label::SUN)
                                    .value("EARTH", UnrealEngine::Label::EARTH)
                                    .value("HUMAN", UnrealEngine::Label::HUMAN);
  unreal_engine.attr("AssetState")
      = py::class_<UnrealEngine::AssetState>(unreal_engine, "AssetState")
            .def(py::init<>())
            .def_readwrite("position", &UnrealEngine::AssetState::position)
            .def_readwrite("velocity", &UnrealEngine::AssetState::velocity)
            .def_readwrite("acceleration", &UnrealEngine::AssetState::acceleration)
            .def_readwrite("orientation", &UnrealEngine::AssetState::orientation)
            .def_readwrite("timestamp", &UnrealEngine::AssetState::timestamp)
            .def("__repr__", [](const UnrealEngine::AssetState& self) {
              return fmt::format(
                  "AssetState(\nposition={}, velocity={}, acceleration={}, orientation={}, "
                  "timestamp={})",
                  self.position.transpose(), self.velocity.transpose(),
                  self.acceleration.transpose(), self.orientation.transpose(), self.timestamp);
            });

  unreal_engine.attr("RenderRequest")
      = py::class_<UnrealEngine::RenderRequest>(unreal_engine, "RenderRequest")
            .def(py::init<>())
            .def_readwrite("agent_id", &UnrealEngine::RenderRequest::agent_id)
            .def_readwrite("camera", &UnrealEngine::RenderRequest::camera)
            .def_readwrite("render_type", &UnrealEngine::RenderRequest::render_type)
            .def_readwrite("width", &UnrealEngine::RenderRequest::width)
            .def_readwrite("height", &UnrealEngine::RenderRequest::height)
            .def_readwrite("fov", &UnrealEngine::RenderRequest::fov);

  unreal_engine.attr("RenderResult")
      = py::class_<UnrealEngine::RenderResult>(unreal_engine, "RenderResult")
            .def(py::init<>())
            .def_readwrite("id", &UnrealEngine::RenderResult::id)
            .def_readwrite("image", &UnrealEngine::RenderResult::image)
            .def_readwrite("width", &UnrealEngine::RenderResult::width)
            .def_readwrite("height", &UnrealEngine::RenderResult::height);
}
