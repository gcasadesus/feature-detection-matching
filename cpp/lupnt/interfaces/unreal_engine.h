#pragma once

#include <Eigen/Dense>
#include <mutex>
#include <opencv2/core.hpp>
#include <string>

#include "lupnt/core/config.h"
#include "lupnt/core/definitions.h"

namespace lupnt {
  // Coordinate conversion utilities
  Vec3 XyzToXyzUnreal(const Vec3& xyz);
  Vec3 XyzUnrealToXyz(const Vec3& xyz_unreal);
  Vec3 RpyToRpyUnreal(const Vec3& rpy);
  Vec3 RpyUnrealToRpy(const Vec3& rpy_unreal);
  Vec3 RotToRpyUnreal(const Mat3& rot);
  Mat3 RpyUnrealToRot(const Vec3& rpy_unreal);

  class UnrealEngine {
  public:
    enum class Label {
      SKY = 0,
      REGOLITH = 1,
      ROVER = 2,
      ROCK = 3,
      LANDER = 4,
      SUN = 5,
      EARTH = 6,
      HUMAN = 7
    };

    static constexpr const char* HOST = "127.0.0.1";
    static constexpr int TCP_PORT = 12345;
    static constexpr int RESPONSE_SIZE = 1024;
    static constexpr int BUFFER_SIZE = 8192;
    static constexpr int METADATA_SIZE = 12;
    static constexpr double TIMEOUT = 10.0;

    static constexpr double REFERENCE_HEIGHT = 1300.0;  // [m]
    static constexpr double MAX_DEPTH = 20.0e3;         // [m]

    static const Vec3 UE_REFERENCE_MOON_PA;

    // State
    struct AssetState {
      Vec3 position;
      Vec3 velocity;
      Vec3 acceleration;
      Vec3 orientation;
      Real timestamp;
    };

    UnrealEngine(const std::string& host = HOST, int port = TCP_PORT);
    ~UnrealEngine();

    void Connect();
    void Disconnect();

    // Communication
    std::string SendRequest(const std::string& msg_json);
    std::string ReceiveResponse();

    // Render/image
    cv::Mat ReceiveRender(const std::string& render_type);

    // Simulation control
    void Pause();
    void Play();
    void Step(double duration);
    void SetTimeDilation(double time_dilation);
    void SetTimestep(double timestep);
    void SetSun(double azimuth, double elevation);

    // Agent
    void Spawn(const std::string& agent_id, const std::string& type, const Vec3& xyz,
               const Vec3& rpy, const Config& config = Config());
    void Spawn(const std::string& agent_id, const std::string& type, const Vec3& xyz,
               const Mat3& rot, const Config& config = Config());
    void SetControl(const std::string& agent_id, Real throttle_value, Real steering_value);
    void SetActorTransform(const std::string& agent_id, const Vec3& xyz, const Vec3& rpy);
    void SetActorTransform(const std::string& agent_id, const Vec3& xyz, const Mat3& rot);
    void SetPitchYaw(const std::string& agent_id, Real pitch_value, Real yaw_value);
    void Remove(const std::string& agent_id);
    void RemoveAll();

    // Getters
    AssetState GetState(const std::string& agent_id);
    std::map<std::string, std::vector<std::string>> GetAgentComponents(const std::string& agent_id);

    // Camera/Rendering
    cv::Mat Render(const std::string& agent_id, const std::string& camera, const std::string& type,
                   std::optional<int32_t> width = std::nullopt,
                   std::optional<int32_t> height = std::nullopt,
                   std::optional<float> fov = std::nullopt,
                   std::optional<float> ss_factor = std::nullopt);

    // Batch rendering with shared memory
    struct RenderRequest {
      std::string agent_id;
      std::string camera;
      std::string render_type;
      std::optional<int32_t> width;
      std::optional<int32_t> height;
      std::optional<float> fov;
      std::optional<float> ss_factor;  // Supersampling factor (default: 3.0, only for RGB)
      std::optional<bool> grayscale;   // Single-channel grayscale (saves 75% bandwidth)
    };

    struct RenderResult {
      std::string id;  // Combined: agent_id/camera/render_type
      cv::Mat image;
      int32_t width;
      int32_t height;
    };

    std::vector<RenderResult> BatchRender(const std::vector<RenderRequest>& requests);
    void InitSharedMemory(const std::string& name = "/lupnt_render_shm", size_t size_mb = 2000);
    void ShutdownSharedMemory();
    void ClearRenderTargetPool();  // Clear render target pool to free GPU memory

    void SetTargetView(const std::string& agent_id);
    void ResetCameraProperties(const std::string& agent_id, const std::string& camera_name);

    // Component Management
    void AddToAgent(const std::string& agent_id, const std::string& type, const std::string& name,
                    const Config& params = Config());
    void RemoveFromAgent(const std::string& agent_id, const std::string& type,
                         const std::string& name);
    void SetComponentTransform(const std::string& agent_id, const std::string& type,
                               const std::string& name, const Vec3& xyz, const Vec3& rpy);
    void SetComponentTransform(const std::string& agent_id, const std::string& type,
                               const std::string& name, const Vec3& xyz, const Mat3& rot);
    void SetComponentProperties(const std::string& agent_id, const std::string& type,
                                const std::string& name, const Config& properties);

  private:
    std::string host_;
    int port_;
    int socket_fd_;
    bool connected_;
    std::recursive_mutex mutex_;
    std::string name_;

    // Shared memory
    int shm_fd_;
    void* shm_ptr_;
    size_t shm_size_;
    std::string shm_name_;

    void ReceiveExactly(void* buffer, size_t size);
    cv::Mat ReadImageFromSharedMemory(size_t offset, int32_t width, int32_t height, size_t size,
                                      const std::string& render_type);

    // Disable copy
    UnrealEngine(const UnrealEngine&) = delete;
    UnrealEngine& operator=(const UnrealEngine&) = delete;
  };
}  // namespace lupnt
