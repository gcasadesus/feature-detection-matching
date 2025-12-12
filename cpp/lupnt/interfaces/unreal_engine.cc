#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstring>  // For memcpy

#if defined(__linux__) || defined(__APPLE__)
#  include <fcntl.h>
#  include <sys/mman.h>
#  include <sys/stat.h>

#  include <Eigen/Dense>
#  include <mutex>
#  include <nlohmann/json.hpp>
#  include <opencv2/opencv.hpp>
#  include <string>
#  include <vector>

#  include "lupnt/conversions/coordinate_conversions.h"
#  include "lupnt/core/config.h"
#  include "lupnt/core/constants.h"
#  include "lupnt/core/definitions.h"
#  include "lupnt/core/error.h"
#  include "lupnt/core/logger.h"
#  include "lupnt/interfaces/unreal_engine.h"
#  include "lupnt/interfaces/yaml.h"
#  include "lupnt/numerics/math_utils.h"

namespace lupnt {

  const Vec3 UnrealEngine::UE_REFERENCE_MOON_PA
      = LatLonAltToCart(Vec3(-90.0, -90.0, REFERENCE_HEIGHT), R_MOON);

  /**
   * @brief Convert position from standard coordinates to Unreal Engine coordinates
   * @param xyz Position in standard coordinates [x, y, z] (m)
   * @return Position in Unreal Engine coordinates [y, x, z] (cm)
   */
  Vec3 XyzToXyzUnreal(const Vec3& xyz) { return Vec3(xyz[0], -xyz[1], xyz[2]) * CM_M; }

  /**
   * @brief Convert position from Unreal Engine coordinates to standard coordinates
   * @param xyz_unreal Position in Unreal Engine coordinates [y, x, z]
   * @return Position in standard coordinates [x, y, z]
   */
  Vec3 XyzUnrealToXyz(const Vec3& xyz_unreal) {
    return Vec3(xyz_unreal[0], -xyz_unreal[1], xyz_unreal[2]) * M_CM;
  }

  /**
   * @brief Convert roll-pitch-yaw to Unreal Engine roll-pitch-yaw
   * @param rpy Roll-pitch-yaw in standard coordinates [roll, pitch, yaw] (radians)
   * @return Roll-pitch-yaw in Unreal Engine coordinates (degrees)
   */
  Vec3 RpyToRpyUnreal(const Vec3& rpy) { return Vec3(rpy[0], -rpy[1], -rpy[2]) * DEG; }

  /**
   * @brief Convert Unreal Engine roll-pitch-yaw to standard roll-pitch-yaw
   * @param rpy_unreal Roll-pitch-yaw in Unreal Engine coordinates (radians)
   * @return Roll-pitch-yaw in standard coordinates [roll, pitch, yaw] (radians)
   */
  Vec3 RpyUnrealToRpy(const Vec3& rpy_unreal) {
    return Vec3(rpy_unreal[0], -rpy_unreal[1], -rpy_unreal[2]) * RAD;
  }

  /**
   * @brief Convert rotation matrix to Unreal Engine roll-pitch-yaw
   * @param rot Rotation matrix (3x3)
   * @return Roll-pitch-yaw in Unreal Engine coordinates (degrees)
   */
  Vec3 RotToRpyUnreal(const Mat3& rot) {
    Vec3 rpy = rot.eulerAngles(2, 1, 0);
    Vec3 rpy_ordered(rpy[2], rpy[1], rpy[0]);
    return RpyToRpyUnreal(rpy_ordered);
  }

  /**
   * @brief Convert Unreal Engine roll-pitch-yaw to rotation matrix
   * @param rpy_unreal Roll-pitch-yaw in Unreal Engine coordinates (radians)
   * @return Rotation matrix (3x3)
   */
  Mat3 RpyUnrealToRot(const Vec3& rpy_unreal) {
    Vec3 rpy = RpyUnrealToRpy(rpy_unreal);
    return (RotX(rpy[0]) * RotY(rpy[1]) * RotZ(rpy[2])).transpose();
  }

  /**
   * @brief Construct a new Unreal Engine interface
   * @param host Server hostname or IP address (default: "127.0.0.1")
   * @param port Server TCP port (default: 12345)
   */
  UnrealEngine::UnrealEngine(const std::string& host, int port)
      : host_(host),
        port_(port),
        socket_fd_(-1),
        connected_(false),
        name_("UnrealEngine"),
        shm_fd_(-1),
        shm_ptr_(nullptr),
        shm_size_(0),
        shm_name_("") {}

  UnrealEngine::~UnrealEngine() {
    ShutdownSharedMemory();
    Disconnect();
  }

  /**
   * @brief Connect to the Unreal Engine server via TCP/IP
   * @throws lupnt::Error if connection fails
   */
  void UnrealEngine::Connect() {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (connected_) return;

    socket_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (socket_fd_ < 0) LUPNT_CHECK(false, "Socket creation failed", "UnrealEngine");

    sockaddr_in serv_addr{};
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(port_);
    if (inet_pton(AF_INET, host_.c_str(), &serv_addr.sin_addr) <= 0)
      LUPNT_CHECK(false, "Invalid address", "UnrealEngine");

    if (connect(socket_fd_, (struct sockaddr*)&serv_addr, sizeof(serv_addr)) < 0)
      LUPNT_CHECK(false, "Connection failed", "UnrealEngine");

    connected_ = true;
    Logger::Debug("Connected " + host_ + ":" + std::to_string(port_), name_);
  }

  /**
   * @brief Disconnect from the Unreal Engine server
   */
  void UnrealEngine::Disconnect() {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (connected_) {
      close(socket_fd_);
      socket_fd_ = -1;
      connected_ = false;
      Logger::Debug("Disconnected", name_);
    }
  }

  /**
   * @brief Send a JSON request to the server and receive the response
   * @param msg_json JSON-formatted request string
   * @return JSON-formatted response string
   * @throws lupnt::Error if send fails or server returns error status
   */
  std::string UnrealEngine::SendRequest(const std::string& msg_json) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (!connected_) LUPNT_CHECK(false, "Not connected", "UnrealEngine");
    ssize_t sent = send(socket_fd_, msg_json.c_str(), msg_json.size(), 0);
    if (sent < 0) LUPNT_CHECK(false, "Send failed", "UnrealEngine");

    Logger::Debug("Sent " + msg_json, name_);
    std::string response = ReceiveResponse();

    json response_json = json::parse(response);
    if (response_json.contains("status") && response_json["status"] == "failed") {
      std::string error_msg = response_json.value("message", "Unknown error");
      LUPNT_CHECK(false, "UnrealEngine request failed: " + error_msg, "UnrealEngine");
    }
    return response;
  }

  /**
   * @brief Receive a response from the server (fixed or variable length)
   * @return JSON-formatted response string
   * @throws lupnt::Error if receive fails
   */
  std::string UnrealEngine::ReceiveResponse() {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (!connected_) LUPNT_CHECK(false, "Not connected", "UnrealEngine");

    // First, read exactly 1024 bytes (minimum response size for both protocols)
    char buffer[RESPONSE_SIZE];
    size_t total_received = 0;
    while (total_received < RESPONSE_SIZE) {
      ssize_t received
          = recv(socket_fd_, buffer + total_received, RESPONSE_SIZE - total_received, 0);
      if (received <= 0) {
        LUPNT_CHECK(false, "Receive failed", "UnrealEngine");
      }
      total_received += received;
    }

    std::string response_json(buffer, RESPONSE_SIZE);

    // Check if it's a complete response (null-padded - old protocol)
    size_t null_pos = response_json.find('\0');
    if (null_pos != std::string::npos) {
      // Fixed-size response (old protocol) - STOP HERE, image data may follow
      response_json = response_json.substr(0, null_pos);
      Logger::Debug("Received fixed-size response: " + response_json, name_);
      return response_json;
    }

    // No null padding - check if we have a complete JSON with newline terminator
    size_t newline_pos = response_json.find('\n');
    if (newline_pos != std::string::npos) {
      std::string candidate = response_json.substr(0, newline_pos);
      try {
        auto _ = json::parse(candidate);  // Try to parse
        Logger::Debug("Received variable-length response (" + std::to_string(candidate.size())
                          + " bytes) - complete in first chunk",
                      name_);
        return candidate;
      } catch (...) {
        // JSON incomplete, continue reading
      }
    }

    // No null and no complete JSON - this is a large response, continue reading
    // The server sends compact JSON terminated by a single newline
    size_t old_newline_pos = (newline_pos == std::string::npos) ? 0 : newline_pos;
    char extra_buffer[BUFFER_SIZE];
    while (true) {
      ssize_t received = recv(socket_fd_, extra_buffer, BUFFER_SIZE, 0);
      if (received <= 0) {
        LUPNT_CHECK(false, "Receive failed", "UnrealEngine");
      }

      response_json.append(extra_buffer, received);

      // Search for newline - use rfind() to get the LAST one
      size_t last_newline = response_json.rfind('\n');
      if (last_newline != std::string::npos && last_newline > old_newline_pos) {
        // Found a NEW newline beyond the first 1024 bytes
        // Check if this makes a valid JSON before returning
        std::string candidate = response_json.substr(0, last_newline);
        try {
          auto _ = json::parse(candidate);
          Logger::Debug("Received variable-length response (" + std::to_string(candidate.size())
                            + " bytes) - valid JSON",
                        name_);
          return candidate;
        } catch (...) {
          // JSON incomplete (internal newline?), continue reading
          old_newline_pos = last_newline;
        }
      }

      // Safety check: prevent infinite loop
      if (response_json.size() > 10 * 1024 * 1024) {  // 10 MB max
        LUPNT_CHECK(false, "Response too large (> 10 MB)", "UnrealEngine");
      }
    }
  }

  /**
   * @brief Receive exact number of bytes from the server
   * @param buffer Buffer to store received data
   * @param size Number of bytes to receive
   * @throws lupnt::Error if receive fails
   */
  void UnrealEngine::ReceiveExactly(void* buffer, size_t size) {
    size_t received = 0;
    char* buf = static_cast<char*>(buffer);
    Logger::Debug("Receiving exactly " + std::to_string(size), name_);
    while (received < size) {
      ssize_t r = recv(socket_fd_, buf + received, size - received, 0);
      if (r <= 0) LUPNT_CHECK(false, "Receive failed", "UnrealEngine");
      received += r;
    }
    Logger::Debug("Received exactly " + std::to_string(received), name_);
  }

  cv::Mat UnrealEngine::ReceiveRender(const std::string& render_type) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (!connected_) LUPNT_CHECK(false, "Not connected", "UnrealEngine");

    uint8_t metadata[METADATA_SIZE];
    ReceiveExactly(metadata, METADATA_SIZE);

    int width = *reinterpret_cast<uint32_t*>(&metadata[0]);
    int height = *reinterpret_cast<uint32_t*>(&metadata[4]);
    int size = *reinterpret_cast<uint32_t*>(&metadata[8]);

    // OPTIMIZATION: Check for grayscale sizing
    bool is_grayscale = (size == (size_t)(width * height));

    // Allocate buffer for TCP read
    std::vector<uint8_t> buffer(size);
    ReceiveExactly(buffer.data(), size);

    cv::Mat image;

    if (is_grayscale) {
      // Direct copy from buffer
      image = cv::Mat(height, width, CV_8UC1);
      std::memcpy(image.data, buffer.data(), size);
    } else if (render_type == "depth") {
      image = cv::Mat(height, width, CV_32FC1);
      float* dst_ptr = reinterpret_cast<float*>(image.data);
      const uint8_t* src_ptr = buffer.data();
      const int num_pixels = width * height;
      const float normalizer = static_cast<float>(MAX_DEPTH) / 16777215.0f;

      // OPTIMIZED: Single pass pointer math loop (No OpenCV overhead)
      // UE5 BGRA -> Depth (R + G*256 + B*65536)
      for (int i = 0; i < num_pixels; ++i) {
        int idx = i * 4;
        float b = src_ptr[idx + 0];
        float g = src_ptr[idx + 1];
        float r = src_ptr[idx + 2];
        dst_ptr[i] = (r + g * 256.0f + b * 65536.0f) * normalizer;
      }
    } else {
      // RGBA Color
      size_t expected_size = (size_t)width * height * 4;
      if (render_type == "label" && size < expected_size) {
        // If buffer is smaller, resizing logic or error handling is needed.
        // For now, we assume the received size is correct for the data present
        // and adjust height to match available data to avoid segfault/garbage.
        // This is a workaround for the mismatch.
        int actual_height = size / (width * 4);
        Logger::Warn("Buffer size mismatch! Expected " + std::to_string(expected_size) + " (for "
                         + std::to_string(width) + "x" + std::to_string(height) + ")" + ", got "
                         + std::to_string(size) + ". Adjusting height to "
                         + std::to_string(actual_height),
                     "UnrealEngine");
        height = actual_height;
      }

      if (size < (size_t)width * height * 4) {
        LUPNT_CHECK(false, "Buffer too small for specified dimensions", "UnrealEngine");
      }

      cv::Mat raw_bgra(height, width, CV_8UC4, buffer.data());

      if (render_type == "label") {
        // OPTIMIZED: Extract Channel 2 (Red) directly
        image = cv::Mat(height, width, CV_8UC1);
        int from_to[] = {2, 0};
        cv::mixChannels(&raw_bgra, 1, &image, 1, from_to, 1);
      } else {
        // Convert to RGB
        cv::cvtColor(raw_bgra, image, cv::COLOR_BGRA2RGB);
      }
    }

    return image;
  }

  Config ConvertToUnreal(const Config& config) {
    Config unreal_config = YAML::Clone(config);
    if (unreal_config["xyz"]) {
      unreal_config["xyz"] = EigenToVector<double>(XyzToXyzUnreal(config["xyz"].as<Vec3>()));
    }
    if (unreal_config["rpy"]) {
      unreal_config["rpy"] = EigenToVector<double>(RpyToRpyUnreal(config["rpy"].as<Vec3>()));
    }
    if (unreal_config["rot"]) {
      unreal_config["rot"] = EigenToVector<double>(RotToRpyUnreal(config["rot"].as<Mat3>()));
    }
    if (unreal_config["ortho_width"]) {
      unreal_config["ortho_width"] = config["ortho_width"].as<double>() * CM_M;
    }
    return unreal_config;
  }

  /**
   * @brief Spawn an actor in the simulation
   */
  void UnrealEngine::Spawn(const std::string& agent_id, const std::string& type, const Vec3& xyz,
                           const Vec3& rpy, const Config& config) {
    json msg;
    json config_json = ConfigToJson(config);
    for (auto& [key, value] : config_json.items()) {
      msg[key] = value;
    }
    msg["cmd"] = "SPAWN";
    msg["id"] = agent_id;
    msg["type"] = type;
    msg["xyz"] = EigenToVector<double>(XyzToXyzUnreal(xyz));
    msg["rpy"] = EigenToVector<double>(RpyToRpyUnreal(rpy));
    SendRequest(msg.dump());
  }

  void UnrealEngine::Spawn(const std::string& agent_id, const std::string& type, const Vec3& xyz,
                           const Mat3& rot, const Config& config) {
    json msg;
    json config_json = ConfigToJson(config);
    for (auto& [key, value] : config_json.items()) {
      msg[key] = value;
    }
    msg["cmd"] = "SPAWN";
    msg["id"] = agent_id;
    msg["type"] = type;
    msg["xyz"] = EigenToVector<double>(XyzToXyzUnreal(xyz));
    msg["rpy"] = EigenToVector<double>(RotToRpyUnreal(rot));
    SendRequest(msg.dump());
  }

  /**
   * @brief Set control inputs for a rover
   */
  void UnrealEngine::SetControl(const std::string& agent_id, Real throttle_value,
                                Real steering_value) {
    json msg = {};
    msg["cmd"] = "SET_CONTROL";
    msg["id"] = agent_id;
    msg["throttle"] = throttle_value.val();
    msg["steering"] = steering_value.val();
    SendRequest(msg.dump());
  }

  /**
   * @brief Add a component (camera, light, apriltag) to an agent
   */
  void UnrealEngine::AddToAgent(const std::string& agent_id, const std::string& type,
                                const std::string& name, const Config& params) {
    Config unreal_config = ConvertToUnreal(params);
    json msg = {{"cmd", "ADD_TO_AGENT"}, {"id", agent_id}, {"type", type}, {"name", name}};

    if (!params.IsNull()) {
      json params_json = ConfigToJson(unreal_config);
      for (auto& [key, value] : params_json.items()) {
        msg[key] = value;
      }
    }
    SendRequest(msg.dump());
  }

  /**
   * @brief Remove a component from an agent
   */
  void UnrealEngine::RemoveFromAgent(const std::string& agent_id, const std::string& type,
                                     const std::string& name) {
    json msg = {{"cmd", "REMOVE_FROM_AGENT"}, {"id", agent_id}, {"type", type}, {"name", name}};
    SendRequest(msg.dump());
  }

  /**
   * @brief Set the transform of a component relative to its parent actor
   */
  void UnrealEngine::SetComponentTransform(const std::string& agent_id, const std::string& type,
                                           const std::string& name, const Vec3& xyz,
                                           const Vec3& rpy) {
    json msg = {};
    msg["cmd"] = "SET_TRANSFORM";
    msg["id"] = agent_id;
    msg["type"] = type;
    msg["name"] = name;
    msg["xyz"] = EigenToVector<double>(XyzToXyzUnreal(xyz));
    msg["rpy"] = EigenToVector<double>(RpyToRpyUnreal(rpy));
    SendRequest(msg.dump());
  }

  /**
   * @brief Set the transform of a component relative to its parent actor
   */
  void UnrealEngine::SetComponentTransform(const std::string& agent_id, const std::string& type,
                                           const std::string& name, const Vec3& xyz,
                                           const Mat3& rot) {
    json msg = {};
    msg["cmd"] = "SET_TRANSFORM";
    msg["id"] = agent_id;
    msg["type"] = type;
    msg["name"] = name;
    msg["xyz"] = EigenToVector<double>(XyzToXyzUnreal(xyz));
    msg["rpy"] = EigenToVector<double>(RotToRpyUnreal(rot));
    SendRequest(msg.dump());
  }

  /**
   * @brief Set properties of a component attached to an agent
   */
  void UnrealEngine::SetComponentProperties(const std::string& agent_id, const std::string& type,
                                            const std::string& name, const Config& properties) {
    Config unreal_config = ConvertToUnreal(properties);
    json msg = {{"cmd", "SET_PROPERTIES"}, {"id", agent_id}, {"type", type}, {"name", name}};
    json props_json = ConfigToJson(unreal_config);

    if (type == "camera") {
      std::string render_type = "rgb";
      if (props_json.contains("render")) {
        render_type = props_json["render"];
        props_json.erase("render");
      }
      msg["render"] = render_type;
    }
    msg["properties"] = props_json;
    SendRequest(msg.dump());
  }

  /**
   * @brief Set the transform of an actor in world coordinates
   * @param agent_id Agent ID
   * @param xyz Position
   * @param rpy Orientation (Roll-Pitch-Yaw)
   */
  void UnrealEngine::SetActorTransform(const std::string& agent_id, const Vec3& xyz,
                                       const Vec3& rpy) {
    json msg = {};
    msg["cmd"] = "SET_AGENT_TRANSFORM";
    msg["id"] = agent_id;
    msg["xyz"] = EigenToVector<double>(XyzToXyzUnreal(xyz));
    msg["rpy"] = EigenToVector<double>(RpyToRpyUnreal(rpy));
    SendRequest(msg.dump());
  }

  /**
   * @brief Set the transform of an actor in world coordinates
   * @param agent_id Agent ID
   * @param xyz Position
   * @param rot Orientation matrix
   */
  void UnrealEngine::SetActorTransform(const std::string& agent_id, const Vec3& xyz,
                                       const Mat3& rot) {
    json msg = {};
    msg["cmd"] = "SET_AGENT_TRANSFORM";
    msg["id"] = agent_id;
    msg["xyz"] = EigenToVector<double>(XyzToXyzUnreal(xyz));
    msg["rpy"] = EigenToVector<double>(RotToRpyUnreal(rot));
    SendRequest(msg.dump());
  }

  /**
   * @brief Set pitch and yaw angles for an actor
   * @param agent_id Agent ID
   * @param pitch_value Pitch angle
   * @param yaw_value Yaw angle
   */
  void UnrealEngine::SetPitchYaw(const std::string& agent_id, Real pitch_value, Real yaw_value) {
    json msg = {};
    msg["cmd"] = "SET_PITCH_YAW";
    msg["id"] = agent_id;
    msg["pitch"] = pitch_value.val();
    msg["yaw"] = yaw_value.val();
    SendRequest(msg.dump());
  }

  /**
   * @brief Render an image from a camera
   * @param agent_id Agent ID
   * @param camera Camera name
   * @param render Render type
   * @param width Image width
   * @param height Image height
   * @param fov Field of view
   * @param ss_factor Supersampling factor
   * @return OpenCV Matrix
   */
  cv::Mat UnrealEngine::Render(const std::string& agent_id, const std::string& camera,
                               const std::string& render, std::optional<int32_t> width,
                               std::optional<int32_t> height, std::optional<float> fov,
                               std::optional<float> ss_factor) {
    json msg = {{"cmd", "RENDER"}, {"id", agent_id}, {"camera", camera}, {"render", render}};
    if (width) msg["width"] = *width;
    if (height) msg["height"] = *height;
    if (fov) msg["fov"] = *fov * DEG;
    if (ss_factor) msg["ss_factor"] = *ss_factor;

    std::lock_guard<std::recursive_mutex> lock(mutex_);
    LUPNT_CHECK(connected_, "Not connected", "UnrealEngine");

    SendRequest(msg.dump());
    return ReceiveRender(render);
  }

  /**
   * @brief Pause the simulation
   */
  void UnrealEngine::Pause() {
    json msg = {{"cmd", "PAUSE"}};
    SendRequest(msg.dump());
  }

  /**
   * @brief Resume the simulation (unpause)
   */
  void UnrealEngine::Play() {
    json msg = {{"cmd", "RESUME"}};
    SendRequest(msg.dump());
  }

  /**
   * @brief Step the simulation forward by a fixed duration
   * @param duration Step duration [s]
   */
  void UnrealEngine::Step(double duration) {
    json msg = {{"cmd", "STEP"}, {"duration", duration}};
    SendRequest(msg.dump());
  }

  /**
   * @brief Set the time dilation factor for the simulation
   * @param time_dilation Time dilation factor
   */
  void UnrealEngine::SetTimeDilation(double time_dilation) {
    json msg = {{"cmd", "SET_TIME_DILATION"}, {"time_dilation", time_dilation}};
    SendRequest(msg.dump());
  }

  /**
   * @brief Set the fixed timestep for deterministic simulation
   * @param timestep Timestep [s]
   */
  void UnrealEngine::SetTimestep(double timestep) {
    json msg = {{"cmd", "SET_TIMESTEP"}, {"timestep", timestep}};
    SendRequest(msg.dump());
  }

  /**
   * @brief Get the state of an actor
   */
  UnrealEngine::AssetState UnrealEngine::GetState(const std::string& agent_id) {
    json msg = {{"cmd", "GET_STATE"}, {"id", agent_id}};
    std::string response = SendRequest(msg.dump());
    auto resp = json::parse(response);
    AssetState state;
    for (int i = 0; i < 3; ++i) {
      state.position[i] = resp["position"][i].get<double>();
      state.velocity[i] = resp["velocity"][i].get<double>();
      state.acceleration[i] = resp["acceleration"][i].get<double>();
      state.orientation[i] = resp["orientation"][i].get<double>();
    }
    state.position = XyzUnrealToXyz(state.position);
    state.velocity = XyzUnrealToXyz(state.velocity);
    state.acceleration = XyzUnrealToXyz(state.acceleration);
    state.orientation = RpyUnrealToRpy(state.orientation);
    state.timestamp = resp["timestamp"].get<double>();
    return state;
  }

  /**
   * @brief Get the components of an agent
   */
  std::map<std::string, std::vector<std::string>> UnrealEngine::GetAgentComponents(
      const std::string& agent_id) {
    json msg = {{"cmd", "GET_AGENT_COMPONENTS"}, {"id", agent_id}};
    std::string response = SendRequest(msg.dump());
    auto resp = json::parse(response);
    std::map<std::string, std::vector<std::string>> components;
    for (auto& [type, names] : resp.items()) {
      components[type] = names.get<std::vector<std::string>>();
    }
    return components;
  }

  /**
   * @brief Remove an actor from the simulation
   */
  void UnrealEngine::Remove(const std::string& agent_id) {
    json msg = {{"cmd", "REMOVE"}, {"id", agent_id}};
    SendRequest(msg.dump());
  }

  /**
   * @brief Remove all spawned actors from the simulation
   */
  void UnrealEngine::RemoveAll() {
    json msg = {{"cmd", "REMOVE_ALL"}};
    SendRequest(msg.dump());
  }

  /**
   * @brief Set the sun direction (directional light)
   */
  void UnrealEngine::SetSun(double azimuth, double elevation) {
    json msg = {};
    msg["cmd"] = "SET_SUN";
    msg["rpy"] = {0.0, -elevation * DEG, 90.0 + azimuth * DEG};
    SendRequest(msg.dump());
  }

  /**
   * @brief Set the player camera view target to a specific actor
   */
  void UnrealEngine::SetTargetView(const std::string& agent_id) {
    json msg = {{"cmd", "SET_VIEW_TARGET"}, {"id", agent_id}};
    SendRequest(msg.dump());
  }

  /**
   * @brief Reset camera properties to default values
   */
  void UnrealEngine::ResetCameraProperties(const std::string& agent_id,
                                           const std::string& camera_name) {
    json msg = {{"cmd", "RESET_CAMERA_PROPERTIES"}, {"id", agent_id}, {"camera", camera_name}};
    SendRequest(msg.dump());
  }

  /**
   * @brief Initialize shared memory for batch rendering
   */
  void UnrealEngine::InitSharedMemory(const std::string& name, size_t size_mb) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);

#  if defined(__linux__) || defined(__APPLE__)
    size_t new_size_bytes = size_mb * 1024 * 1024;

    // OPTIMIZATION: Check if we already have this SHM open
    if (shm_ptr_ != nullptr && shm_name_ == name && shm_size_ == new_size_bytes && shm_fd_ != -1) {
      // Re-send init command to Unreal (just in case it restarted), but skip local mmap
      json msg = {{"cmd", "INIT_SHARED_MEMORY"}, {"name", name}, {"size", shm_size_}};
      SendRequest(msg.dump());
      Logger::Info("Shared memory reused: " + name + " (kept open)", name_);
      return;
    }

    if (shm_ptr_ != nullptr) {
      ShutdownSharedMemory();
    }

    shm_name_ = name;
    shm_size_ = new_size_bytes;

    json msg = {{"cmd", "INIT_SHARED_MEMORY"}, {"name", name}, {"size", shm_size_}};
    SendRequest(msg.dump());

    shm_fd_ = shm_open(name.c_str(), O_RDONLY, 0666);
    if (shm_fd_ == -1) {
      LUPNT_CHECK(false, "Failed to open shared memory: " + std::string(strerror(errno)),
                  "UnrealEngine");
    }

    shm_ptr_ = mmap(nullptr, shm_size_, PROT_READ, MAP_SHARED, shm_fd_, 0);
    if (shm_ptr_ == MAP_FAILED) {
      close(shm_fd_);
      shm_fd_ = -1;
      shm_ptr_ = nullptr;
      LUPNT_CHECK(false, "Failed to map shared memory: " + std::string(strerror(errno)),
                  "UnrealEngine");
    }

    Logger::Info("Shared memory initialized: " + name + " (" + std::to_string(size_mb) + " MB)",
                 name_);
#  else
    LUPNT_CHECK(false, "Shared memory only supported on Linux/Mac platforms", "UnrealEngine");
#  endif
  }

  /**
   * @brief Clear render target pool
   */
  void UnrealEngine::ClearRenderTargetPool() {
    json msg = {{"cmd", "CLEAR_POOL"}};
    SendRequest(msg.dump());
    Logger::Info("Cleared render target pool", name_);
  }

  /**
   * @brief Shutdown shared memory
   */
  void UnrealEngine::ShutdownSharedMemory() {
    std::lock_guard<std::recursive_mutex> lock(mutex_);

#  if defined(__linux__) || defined(__APPLE__)
    if (shm_ptr_ != nullptr && shm_ptr_ != MAP_FAILED) {
      munmap(shm_ptr_, shm_size_);
      shm_ptr_ = nullptr;
    }

    if (shm_fd_ != -1) {
      close(shm_fd_);
      shm_fd_ = -1;
    }

    if (!shm_name_.empty() && connected_) {
      json msg = {{"cmd", "SHUTDOWN_SHARED_MEMORY"}};
      try {
        SendRequest(msg.dump());
      } catch (...) {
      }
      Logger::Debug("Shared memory shut down: " + shm_name_, name_);
      shm_name_.clear();
    }
    shm_size_ = 0;
#  endif
  }

  /**
   * @brief Read an image from shared memory (HIGH PERFORMANCE OPTIMIZED)
   * @param offset Byte offset in shared memory
   * @param width Image width
   * @param height Image height
   * @param size Number of bytes
   * @param render_type Type of render ("rgb", "depth", "label")
   * @return OpenCV Mat containing the image (No Clone!)
   */
  cv::Mat UnrealEngine::ReadImageFromSharedMemory(size_t offset, int32_t width, int32_t height,
                                                  size_t size, const std::string& render_type) {
    if (shm_ptr_ == nullptr) {
      LUPNT_CHECK(false, "Shared memory not initialized", "UnrealEngine");
    }

    // Direct pointer to shared memory (Zero Copy from OS perspective)
    const uint8_t* src_ptr = static_cast<const uint8_t*>(shm_ptr_) + offset;

    if (offset + size > shm_size_) {
      LUPNT_CHECK(false, "Read would overflow shared memory bounds", "UnrealEngine");
    }

    cv::Mat image;

    // 1. Grayscale Check (Size Match)
    // If the data size matches w*h exactly, it is already single channel (G8)
    // Or if the user explicitly requested grayscale via name
    bool is_grayscale_format = (size == (size_t)(width * height));

    if (is_grayscale_format) {
      // Create a wrapper header around shared memory (Zero Alloc)
      cv::Mat shm_wrapper(height, width, CV_8UC1, const_cast<uint8_t*>(src_ptr));

      // Perform the ONE required copy from Shared Memory to Local Memory
      // We must copy because SHM is read-only or volatile
      shm_wrapper.copyTo(image);
    }
    // 2. Depth Check
    else if (render_type == "depth") {
      image = cv::Mat(height, width, CV_32FC1);
      float* dst_ptr = reinterpret_cast<float*>(image.data);
      const float normalizer = static_cast<float>(MAX_DEPTH) / 16777215.0f;
      const int num_pixels = width * height;

      // OPTIMIZED LOOP: Single pass pointer arithmetic
      // Avoids cv::split (3 allocs) + convert (1 alloc) + math (1 alloc)
      // UE5 BGRA -> Depth: R + G*256 + B*65536
      for (int i = 0; i < num_pixels; ++i) {
        int idx = i * 4;
        float b = src_ptr[idx + 0];
        float g = src_ptr[idx + 1];
        float r = src_ptr[idx + 2];
        dst_ptr[i] = (r + g * 256.0f + b * 65536.0f) * normalizer;
      }
    }
    // 3. RGBA / Label Check
    else {
      // Wrapper around SHM (Zero Alloc)
      cv::Mat shm_bgra(height, width, CV_8UC4, const_cast<uint8_t*>(src_ptr));

      if (render_type == "label") {
        // UE5 Labels are usually in the Red channel
        // src[2] is Red in BGRA
        image = cv::Mat(height, width, CV_8UC1);
        int from_to[] = {2, 0};
        cv::mixChannels(&shm_bgra, 1, &image, 1, from_to, 1);
      } else {
        // Standard Color: Convert BGRA (SHM) -> RGB (Local)
        // This performs the ONE required copy+convert
        cv::cvtColor(shm_bgra, image, cv::COLOR_BGRA2RGB);
      }
    }

    // cv::Mat is a smart pointer, returning it is cheap.
    return image;
  }

  /**
   * @brief Batch render multiple cameras at once using shared memory
   */
  std::vector<UnrealEngine::RenderResult> UnrealEngine::BatchRender(
      const std::vector<RenderRequest>& requests) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);

    if (shm_ptr_ == nullptr) {
      LUPNT_CHECK(false, "Shared memory not initialized. Call InitSharedMemory() first.",
                  "UnrealEngine");
    }

    json msg = {{"cmd", "BATCH_RENDER"}};
    json requests_array = json::array();

    for (const auto& req : requests) {
      json req_json = {{"id", req.agent_id}, {"camera", req.camera}, {"render", req.render_type}};
      if (req.width) req_json["width"] = *req.width;
      if (req.height) req_json["height"] = *req.height;
      if (req.fov) req_json["fov"] = *req.fov * DEG;
      if (req.ss_factor) req_json["ss_factor"] = *req.ss_factor;
      if (req.grayscale) req_json["grayscale"] = *req.grayscale;
      requests_array.push_back(req_json);
    }
    msg["requests"] = requests_array;

    Logger::Debug("Sending BATCH_RENDER with " + std::to_string(requests.size()) + " requests",
                  name_);

    std::string response = SendRequest(msg.dump());
    auto resp = json::parse(response);

    std::vector<RenderResult> results;
    results.reserve(resp["images"].size());  // Reserve to avoid realloc

    for (const auto& img_meta : resp["images"]) {
      RenderResult result;
      result.id = img_meta["id"].get<std::string>();
      result.width = img_meta["width"].get<int32_t>();
      result.height = img_meta["height"].get<int32_t>();
      size_t offset = img_meta["offset"].get<size_t>();
      size_t size = img_meta["size"].get<size_t>();

      std::string render_type;
      size_t last_slash = result.id.rfind('/');
      if (last_slash != std::string::npos) {
        render_type = result.id.substr(last_slash + 1);
      }

      // Calls the optimized reader
      result.image
          = ReadImageFromSharedMemory(offset, result.width, result.height, size, render_type);
      results.push_back(result);
    }

    Logger::Debug("Batch rendered " + std::to_string(results.size()) + " images", name_);
    return results;
  }

}  // namespace lupnt

#endif  // defined(__linux__) || defined(__APPLE__)
