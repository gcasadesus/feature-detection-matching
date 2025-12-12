import pylupnt as pnt
import numpy as np
import shutil
from pathlib import Path
import yaml
import cv2
import time

pnt.Logger.set_level(pnt.Logger.INFO)


def main():
    # Load DEM
    xlims_enu = [-10e3, 0e3]  # [m]
    ylims_enu = [0, 10e3]  # [m]
    res_enu = 5.0  # [m]
    dem_enu = pnt.load_dem_enu(xlims_enu, ylims_enu, res_enu)

    # Location
    xy_lander = np.array([-7500.0 - 15.0, 4500.0 + 15.0])  # [m]

    # Trajectory 1
    ds = 1.0  # [m]
    R = 20.0  # [m]
    dR = 5.0  # [m]
    theta_traj_tot = 1 * np.pi
    theta_traj = np.linspace(0.0, theta_traj_tot, int(theta_traj_tot * R / ds))
    r = R - dR * np.sin(theta_traj * 8.0) + 0.0 * np.linspace(0.0, 2.0 * dR, len(theta_traj))
    xy_0_rover_0 = xy_lander + np.column_stack([r * np.sin(theta_traj), -r * np.cos(theta_traj)])

    xyz = np.column_stack([xy_0_rover_0 - xy_lander, np.zeros(len(xy_0_rover_0))])
    xyz = xyz @ pnt.rot_z(-90 * pnt.RAD).T
    xy_0_rover_1 = xyz[:, :2] + xy_lander

    # Trajectory 2
    xy_start = xy_lander + np.array([-10.0, -10.0])
    xy_end = xy_lander + np.array([200, 1e3])
    xy_1_rover_0 = np.linspace(xy_start, xy_end, int(np.linalg.norm(xy_end - xy_start) / ds))
    front = (xy_end - xy_start) / np.linalg.norm(xy_end - xy_start)
    left = np.array([-front[1], front[0]])
    length_rover_2 = np.sum(np.sqrt(np.sum(np.diff(xy_1_rover_0, axis=0) ** 2, axis=1)))
    xy_1_rover_0 += (
        10.0
        * left
        * np.sin(np.linspace(0.0, 2.0 * np.pi * length_rover_2 / 50.0, len(xy_1_rover_0)))[:, None]
    )

    xyz = np.column_stack([xy_1_rover_0 - xy_lander, np.zeros(len(xy_1_rover_0))])
    xyz = xyz @ pnt.rot_z(-5 * pnt.RAD).T
    xy_1_rover_1 = xyz[:, :2] + xy_lander + np.array([20.0, 20.0])

    run_config_base = {"az": 180.0, "el": 0.0, "overwrite": True}

    run_configs = {
        "base": {},
        "no_lights": {"enable_lights": []},
        "camera_effects": {
            "camera_properties": {
                # Depth of Field
                "depth_of_field_focal_distance": 200.0,
                "depth_of_field_fstop": 1.0,
                # Motion Blur
                # "enable_motion_blur": True,
                # "motion_blur_amount": 0.25,
                # Chromatic Aberration
                "scene_fringe_intensity": 2.0,
                # Vignette
                # "vignette_intensity": 0.75,
                # Bloom / Lens Flare
                "bloom_intensity": 2.0,
                "lens_flare_intensity": 0.2,
            },
        },
        "higher_elevation": {"az": 180.0, "el": 45.0, "enable_lights": []},
    }

    trajectories = {
        "short": (xy_0_rover_0, xy_0_rover_1),
        "long": (xy_1_rover_0, xy_1_rover_1),
    }

    for traj_name, (xy_rover_0, xy_rover_1) in trajectories.items():
        for run_name, run_config in run_configs.items():
            run_config_tmp = {**run_config_base, **run_config}
            dataset_path = Path(
                f"/home/shared_ws6/data/unreal_engine/local_traverse_fov90/{traj_name}_{run_name}"
            )
            collect_trajectory(
                xy_lander, xy_rover_0, xy_rover_1, dem_enu, dataset_path, run_config_tmp
            )
            with open(dataset_path / "run_config.yaml", "w") as f:
                yaml.dump(run_config_tmp, f)


def calculate_control(
    current_pos,
    velocity_vec,
    xy_trajectory,
    last_steering,
    lookahead_dist=2.0,
    steering_gain=0.7,
    max_steering=0.5,
):
    """Calculate throttle and steering for a rover following a trajectory."""
    THROTTLE_CONST = 0.4

    # Estimate Current Yaw
    if np.linalg.norm(velocity_vec) > 0.1:
        current_yaw = np.arctan2(velocity_vec[1], velocity_vec[0])
        is_moving = True
    else:
        current_yaw = 0.0
        is_moving = False

    # Find the index of the closest point on the path to the rover
    distances = np.linalg.norm(xy_trajectory - current_pos, axis=1)
    closest_idx = np.argmin(distances)

    # Search forward from that closest point to find the first point
    # that is at least lookahead_dist away.
    target_idx = closest_idx
    while target_idx < len(xy_trajectory) - 1:
        dist_to_target = np.linalg.norm(xy_trajectory[target_idx] - current_pos)
        if dist_to_target >= lookahead_dist:
            break
        target_idx += 1

    target_waypoint = xy_trajectory[target_idx]

    # Calculate Steering Angle
    if is_moving:
        # Vector from rover to target in the global (world) frame
        vector_global = target_waypoint - current_pos

        # Transform this vector into the rover's local reference frame
        x_local = vector_global[0] * np.cos(current_yaw) + vector_global[1] * np.sin(current_yaw)
        y_local = -vector_global[0] * np.sin(current_yaw) + vector_global[1] * np.cos(current_yaw)

        # Calculate the angle 'alpha' to the target point in the rover's frame
        alpha = np.arctan2(y_local, x_local)

        # Use a simple proportional controller: steering = gain * error
        steering = -steering_gain * alpha

        # Clip the steering command to the valid range
        steering = np.clip(steering, -max_steering, max_steering)
        last_steering = steering
    else:
        # If not moving, just use the last calculated steering
        steering = last_steering

    throttle = THROTTLE_CONST
    return throttle, steering, last_steering, target_idx


def setup_agent_directories(dataset_path, agent_name, config, dt):
    """Setup directory structure and initialize data collection for an agent."""
    enable_cameras = config["enable_cameras"]
    render_types = ["rgb", "depth", "label"]

    agent_path = dataset_path / agent_name
    gt_path = agent_path / f"state_groundtruth_estimate_{agent_name}"
    imu_path = agent_path / f"imu_{agent_name}"

    # Create directories and CSV files for each camera
    camera_data = {}
    for camera_name in enable_cameras:
        camera_config = config["cameras"][camera_name]
        camera_dir = agent_path / f"cam_{camera_name}"

        # Create directories
        for render_type in render_types:
            (camera_dir / render_type).mkdir(parents=True, exist_ok=True)

        # Open CSV file
        csv_path = camera_dir / "data.csv"
        csv_file = open(csv_path, "w")
        csv_file.write("#timestamp [ns],filename\n")

        # Body to camera transform
        xyz_cam_body = np.array(camera_config["xyz"])
        rpy_cam_body = np.deg2rad(camera_config["rpy"])
        R_cam_body = pnt.roll_pitch_yaw_to_rot(rpy_cam_body)
        body_T_cam = np.eye(4)
        body_T_cam[:3, :3] = R_cam_body
        body_T_cam[:3, 3] = xyz_cam_body

        # Store camera data
        camera_data[camera_name] = {
            "config": camera_config,
            "csv_file": csv_file,
            "camera_dir": camera_dir,
            "body_T_cam": body_T_cam,
            "xyz_cam_body": xyz_cam_body,
            "R_cam_body": R_cam_body,
            "frame_count": 0,
        }

    gt_path.mkdir(parents=True, exist_ok=True)
    imu_path.mkdir(parents=True, exist_ok=True)

    return {
        "agent_path": agent_path,
        "gt_path": gt_path,
        "imu_path": imu_path,
        "camera_data": camera_data,
        "all_timestamps_ns": [],
        "all_body_positions": [],
        "all_body_rotations": [],
        "last_steering": 0.0,
    }


def collect_trajectory(xy_lander, xy_rover_0, xy_rover_1, dem_enu, dataset_path, run_config):
    # Config
    LOOKAHEAD_DIST = 2.0
    STEERING_GAIN = 0.7
    STOP_THRESHOLD = 2.0  # [m]
    LOOP_SLEEP = 0.1  # [s]
    MAX_TIME = 5.0 * pnt.SECS_HOUR  # [s]
    MAX_STEERING = 0.5

    H_free = 768
    W_free = 1024

    # Dataset config
    dt = LOOP_SLEEP

    # Load rover configuration
    config = pnt.load_config("unreal_engine_agents.yaml::rover")
    config["enable_cameras"] = ["front_left", "front_right"]
    enable_cameras = config["enable_cameras"]

    pnt.Logger.info(f"Setting up directory structure at: {dataset_path}", "Main")
    if dataset_path.exists():
        if "overwrite" in run_config and run_config["overwrite"]:
            shutil.rmtree(dataset_path)
        else:
            pnt.Logger.info(f"Dataset path {dataset_path} already exists. Skipping...", "Main")
            return

    # Setup directory structure for both rovers
    rover_data = {
        "rover_0": setup_agent_directories(dataset_path, "rover_0", config, dt),
        "rover_1": setup_agent_directories(dataset_path, "rover_1", config, dt),
    }

    # Setup directory structure for free camera
    free_camera_dir = dataset_path / "free" / "cam_front"
    free_rgb_dir = free_camera_dir / "rgb"
    free_rgb_dir.mkdir(parents=True, exist_ok=True)
    free_csv_path = free_camera_dir / "data.csv"
    free_csv_file = open(free_csv_path, "w")
    free_csv_file.write("#timestamp [ns],filename\n")

    # Trajectories for each rover
    trajectories = {"rover_0": xy_rover_0, "rover_1": xy_rover_1}

    t = 0.0
    t_start = time.time()
    free_camera_frame_count = 0
    with pnt.UnrealEngine() as ue:
        ue.remove_all()
        ue.clear_render_target_pool()
        ue.play()

        # Sun
        az, el = run_config["az"] * pnt.RAD, run_config["el"] * pnt.RAD
        ue.set_sun(az, el)

        # Lander
        z_lander = 1.5 + pnt.interpolate_dem(xy_lander, dem_enu)
        xyz_lander = np.concatenate([xy_lander, [z_lander]])
        R_lander = np.eye(3)
        ue.spawn("lander", "blue_moon_lander", xyz_lander, R_lander)

        # Spawn both rovers
        for rover_name, xy_traj in trajectories.items():
            z_rover = 0.5 + pnt.interpolate_dem(xy_traj[0], dem_enu)
            xyz_rover = np.concatenate([xy_traj[0], [z_rover]])
            xyz_rover_next = np.concatenate([xy_traj[1], [z_rover]])
            R_rover = pnt.look_at(xyz_rover, xyz_rover_next)
            ue.spawn(rover_name, "opportunity_rover", xyz_rover, R_rover)

            # Add lights to rover
            if "enable_lights" in run_config:
                enable_lights = run_config["enable_lights"]
            else:
                enable_lights = config["enable_lights"]
            for light_name in enable_lights:
                light_config = config["lights"][light_name].copy()
                light_config["rpy"] = np.deg2rad(light_config["rpy"])
                ue.add_to_agent(rover_name, "light", light_name, light_config)

            # Add cameras to rover
            for cam_name in config["enable_cameras"]:
                cam_config = config["cameras"][cam_name].to_dict()
                cam_config["rpy"] = np.deg2rad(cam_config["rpy"])
                if "camera_properties" in run_config:
                    cam_config.update(**run_config["camera_properties"])
                ue.add_to_agent(rover_name, "camera", cam_name, cam_config)

        # Free camera
        xyz_free = xyz_lander + np.array([-25.0, -20.0, 20.0])  # [m]
        R_free = pnt.look_at(xyz_free, xyz_lander)
        ue.spawn("free", "free_agent", xyz_free, R_free)
        ue.add_to_agent("free", "camera", "front", {"width": 1024, "height": 768, "fov": 60.0})

        # Screen
        ue.set_target_view("free")

        # Time
        ue.set_time_dilation(1.0)
        ue.set_timestep(1.0 / 20.0)

        pnt.Logger.info("Waiting for 5 seconds", "Main")
        time.sleep(5)

        # Do two render passes to warm up the cache, with progress bar
        n_iters = 5
        n_cams = sum(len(rover_data[rover]["camera_data"]) for rover in ["rover_0", "rover_1"])
        pbar = pnt.Logger.get_progress_bar(n_iters * n_cams, "Warming up", "Main")
        for _ in range(n_iters):
            for rover_name in ["rover_0", "rover_1"]:
                for cam_name, cam_data in rover_data[rover_name]["camera_data"].items():
                    W, H = cam_data["config"]["width"], cam_data["config"]["height"]
                    ue.render(rover_name, cam_name, "rgb", W, H)
                    ue.render(rover_name, cam_name, "depth", W, H)
                    ue.render(rover_name, cam_name, "label", W, H)
                    pbar.update()
            ue.render("free", "front", "rgb", W_free, H_free)
        pbar.finish()

        ue.pause()

        pbar = pnt.Logger.get_progress_bar(100, "Collecting trajectory", "Main")
        pbar_val = 0
        pbar_val_prev = 0

        # Track progress for each rover
        target_indices = {"rover_0": 0, "rover_1": 0}
        rovers_finished = {"rover_0": False, "rover_1": False}

        while t < MAX_TIME:
            # Calculate timestamp
            timestamp_ns = int(t * 1e9)

            # Process each rover
            for rover_name in ["rover_0", "rover_1"]:
                if rovers_finished[rover_name]:
                    continue

                data = rover_data[rover_name]
                xy_traj = trajectories[rover_name]

                # Get Current Rover State
                state = ue.get_state(rover_name)
                current_pos = state.position[:2]
                velocity_vec = state.velocity[:2]

                # Get rover state
                xyz_body = state.position
                rpy_body = state.orientation
                R_body = pnt.roll_pitch_yaw_to_rot(rpy_body)

                # Store body pose
                data["all_timestamps_ns"].append(timestamp_ns)
                data["all_body_positions"].append(xyz_body.copy())
                data["all_body_rotations"].append(R_body.copy())

                for camera_name, cam_data in data["camera_data"].items():
                    W, H = cam_data["config"]["width"], cam_data["config"]["height"]
                    frame_count = cam_data["frame_count"]
                    camera_dir = cam_data["camera_dir"]
                    csv_file = cam_data["csv_file"]

                    # Render RGB
                    rgb_path = camera_dir / "rgb" / f"{frame_count:06d}.png"
                    rgb_image = ue.render(rover_name, camera_name, "rgb", W, H)
                    cv2.imwrite(str(rgb_path), cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
                    csv_file.write(f"{timestamp_ns},{frame_count:06d}.png\n")

                    # Render Depth
                    depth_path = camera_dir / "depth" / f"{frame_count:06d}.tiff"
                    depth_image = ue.render(rover_name, camera_name, "depth", W, H)
                    cv2.imwrite(str(depth_path), depth_image)

                    # Render Label
                    label_path = camera_dir / "label" / f"{frame_count:06d}.png"
                    label_image = ue.render(rover_name, camera_name, "label", W, H)
                    cv2.imwrite(str(label_path), label_image)

                    cam_data["frame_count"] += 1

                # Check for Termination
                final_goal = xy_traj[-1]
                dist_to_goal = np.linalg.norm(current_pos - final_goal)
                if dist_to_goal < STOP_THRESHOLD:
                    pnt.Logger.info(f"{rover_name} reached final goal!", "Main")
                    ue.set_control(rover_name, 0.0, 0.0)
                    rovers_finished[rover_name] = True
                    continue

                # Calculate control
                throttle, steering, data["last_steering"], target_idx = calculate_control(
                    current_pos,
                    velocity_vec,
                    xy_traj,
                    data["last_steering"],
                    LOOKAHEAD_DIST,
                    STEERING_GAIN,
                    MAX_STEERING,
                )
                target_indices[rover_name] = target_idx

                # Set Controls
                ue.set_control(rover_name, throttle, steering)

            # Render and save from free camera
            free_rgb = ue.render("free", "front", "rgb", W_free, H_free)
            free_rgb_path = free_rgb_dir / f"{free_camera_frame_count:06d}.png"
            cv2.imwrite(str(free_rgb_path), cv2.cvtColor(free_rgb, cv2.COLOR_RGB2BGR))
            free_csv_file.write(f"{timestamp_ns},{free_camera_frame_count:06d}.png\n")
            free_camera_frame_count += 1

            # Update progress bar with per-rover percentages and realtime factor
            rover_percents = {}
            for r in ["rover_0", "rover_1"]:
                rover_percents[r] = 100 * target_indices[r] / max(1, len(trajectories[r]))
            avg_progress = sum(rover_percents.values()) / (2 * 100)
            pbar_val = int(max(t / MAX_TIME, avg_progress) * 100)

            realtime_factor = (t / (time.time() - t_start)) if (time.time() - t_start) > 0 else 0
            desc = (
                f"rover_0: {rover_percents['rover_0']:.1f}%, "
                f"rover_1: {rover_percents['rover_1']:.1f}%, "
                f"realtime: {realtime_factor:.2f}x"
            )
            pbar.set_description(desc)
            if pbar_val > pbar_val_prev and pbar_val < 100:
                pbar.update(pbar_val)
                pbar_val_prev = pbar_val

            # Check if all rovers are done
            if all(rovers_finished.values()):
                pnt.Logger.info("All rovers finished!", "Main")
                break

            # Step simulation once for all rovers
            ue.step(LOOP_SLEEP)
            t += LOOP_SLEEP

    t_end = time.time()
    realtime_factor = t / (t_end - t_start)
    pnt.Logger.info(f"Total time: {t_end - t_start} s", "Main")
    pnt.Logger.info(f"Realtime factor: {realtime_factor:.2f}x", "Main")

    pbar.finish()

    # Close free camera CSV file
    free_csv_file.close()

    # Post-process and write ground truth and calibration files
    pnt.Logger.info("Writing groundtruth and calibration files...", "Main")

    class BracketedListDumper(yaml.SafeDumper):
        def represent_list(self, data):
            return self.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)

    BracketedListDumper.add_representer(list, BracketedListDumper.represent_list)

    # Process each rover's data
    for rover_name, data in rover_data.items():
        # Close all CSV files
        for cam_data in data["camera_data"].values():
            cam_data["csv_file"].close()

        # Write ground truth file for the body
        all_body_positions = np.array(data["all_body_positions"])
        all_body_rotations = np.array(data["all_body_rotations"])
        all_timestamps_ns = data["all_timestamps_ns"]

        with open(data["gt_path"] / "data.csv", "w") as f:
            f.write(
                "#timestamp [ns], "
                "p_x [m], p_y [m], p_z [m], "
                "q_w, q_x, q_y, q_z, "
                "v_x [m/s], v_y [m/s], v_z [m/s], "
                "b_w_x [rad/s], b_w_y [rad/s], b_w_z [rad/s], "
                "b_a_x [m/s^2], b_a_y [m/s^2], b_a_z [m/s^2]\n"
            )
            velocities = np.gradient(all_body_positions, dt, axis=0)
            zero_biases = np.zeros(6)

            for i in range(len(all_timestamps_ns)):
                # Rotation matrix should already be 3x3
                R = all_body_rotations[i]
                quat_wxyz = pnt.rot_to_quat(R)
                data_list = (
                    [all_timestamps_ns[i]]
                    + all_body_positions[i].tolist()
                    + quat_wxyz.tolist()
                    + velocities[i].tolist()
                    + zero_biases.tolist()
                )
                f.write(",".join(map(str, data_list)) + "\n")

        # Write YAML Calibration Files for each camera
        for camera_name, cam_data in data["camera_data"].items():
            camera_config = cam_data["config"]
            W = camera_config["width"]
            H = camera_config["height"]
            fov = camera_config["fov"] * pnt.RAD
            fx, fy = pnt.fov_to_focal_length([W, H], [fov, fov])
            cx = W / 2.0
            cy = H / 2.0
            intrinsics = [fx, fy, cx, cy]

            sensor_yaml = {
                "sensor_type": "camera",
                "comment": f"{camera_name.capitalize()} camera",
                "T_BS": {
                    "rows": 4,
                    "cols": 4,
                    "data": [float(val) for val in cam_data["body_T_cam"].flatten()],
                },
                "rate_hz": 1.0 / dt,
                "resolution": [W, H],
                "camera_model": "pinhole",
                "intrinsics": [float(val) for val in intrinsics],
                "distortion_model": "radtan",
                "distortion_coeffs": [0.0, 0.0, 0.0, 0.0],
            }

            with open(cam_data["camera_dir"] / "sensor.yaml", "w") as f:
                yaml.dump(sensor_yaml, f, sort_keys=False, Dumper=BracketedListDumper)

    pnt.Logger.info(f"Dataset creation complete at: {dataset_path}", "Main")
    pnt.Logger.info(f"Total frames saved: {frame_count}", "Main")
    pnt.Logger.info(f"Cameras processed: {', '.join(enable_cameras)}", "Main")


if __name__ == "__main__":
    main()
