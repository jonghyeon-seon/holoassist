import os
import numpy as np
import cv2
from sklearn.neighbors import NearestNeighbors
from fractions import Fraction
import torch
from torchvision.datasets.vision import VisionDataset
import einops
from scipy.spatial.transform import Rotation as R
import random

# --- Coordinate transformation matrices ---
axis_transform = np.linalg.inv(
    np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
)

axis2_transform = np.array([[-1, 0, 0, 0], [0, 0, -1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])


# --- IMU utilities ---
def read_imu_file(path):
    """
    Reads a sync text with format:
      rel_time  abs_timestamp  x  y  z
    Returns:
      rel_times: (N,) float seconds
      abs_times: (N,) float (nanoseconds)
      vals:      (N,3) float sensor readings
    """
    arr = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split()
            rel = float(parts[0])
            ts = float(parts[1])
            vals = list(map(float, parts[2:5]))
            arr.append([rel, ts] + vals)
    data = np.array(arr, dtype=np.float64)
    return data[:, 0], data[:, 1], data[:, 2:5]


def integrate_gyro(rel_times, gyro_vals):
    """
    Integrate gyroscope rates to rotation matrices.
    rel_times: (N,) seconds
    gyro_vals: (N,3) rad/s
    Returns:
      rotations: list of (3,3) rotation matrices, length N
    """
    N = len(rel_times)
    rotations = [np.eye(3, dtype=np.float64)]
    for i in range(1, N):
        dt = rel_times[i] - rel_times[i - 1]
        omega = gyro_vals[i] * dt  # small-angle
        dR = R.from_rotvec(omega).as_matrix()
        R_i = rotations[-1] @ dR
        rotations.append(R_i)
    return rotations


def integrate_imu_gyro(rel_times, acc_vals, gyro_vals):
    """
    Double-integrate accelerometer in world frame, using gyro-integrated rotations.
    Returns:
      positions: (N,3)
      rotations: list of (3,3)
    """
    rotations = integrate_gyro(rel_times, gyro_vals)
    g = np.array([0, 0, 9.81], dtype=np.float64)
    N = len(rel_times)
    v = np.zeros(3, dtype=np.float64)
    p = np.zeros(3, dtype=np.float64)
    positions = np.zeros((N, 3), dtype=np.float64)
    positions[0] = p
    for i in range(1, N):
        dt = rel_times[i] - rel_times[i - 1]
        R_i = rotations[i]
        acc_world = R_i @ acc_vals[i] - g
        v += acc_world * dt
        p += v * dt
        positions[i] = p
    return positions, rotations


# --- File reading utilities ---
class IndexSearch:
    def __init__(self, time_array):
        self.time_array = time_array
        self.prev = 0
        self.index = 0
        self.len = len(time_array)

    def nearest_neighbor(self, target_time):
        while target_time > self.time_array[self.index]:
            if self.len - 1 <= self.index:
                return self.index
            self.index += 1
            self.prev = self.time_array[self.index]

        if (
            abs(self.time_array[self.index] - target_time)
            > abs(self.time_array[self.index - 1] - target_time)
        ) and (self.index != 0):
            ret_index = self.index - 1
        else:
            ret_index = self.index
        return ret_index


def read_pose_sync_txt(path):
    """Read Timing_sync.txt, returning extrinsic lines in index order."""
    return [l for l in open(path).read().splitlines() if l.strip()]


def read_pose_txt(path):
    """Read pose text file with tab-separated values."""
    arr = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                break
            arr.append(list(map(float, line.split("\t"))))
    return np.array(arr)


def read_hand_pose_txt(path):
    """Read hand pose text file with tab-separated values."""
    arr = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                break
            vals = list(map(float, line.split("\t")))
            mats = np.reshape(vals[3:-52], (-1, 4, 4))
            xyz = [(m @ np.array([0, 0, 0, 1]).T)[:3] for m in mats]
            row = vals[:4] + list(np.concatenate(xyz))
            arr.append(row)
    return np.array(arr)


def read_intrinsics_txt(path):
    """Read camera intrinsics from text file."""
    vals = list(map(float, open(path).read().split("\t")))
    return np.array(vals[:9]).reshape(3, 3)


def read_depth_intrinsics_txt(path):
    """Read depth camera intrinsics from text file."""
    vals = list(map(float, open(path).read().split("\t")))
    return np.array(vals[:14])


def read_timing_sync_txt(path):
    """Read timing sync text file, returning index to timestamp mapping."""
    index_to_time = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            vals = line.strip().split()
            idx = int(vals[1])
            timestamp = float(vals[2])
            index_to_time[idx] = timestamp
    return index_to_time


# --- Video and timing utilities ---
def get_video_frame_times(video_path, timing_txt_path):
    """Calculate timestamps for each frame in a video."""
    with open(timing_txt_path) as f:
        start_time = int(f.readline().strip())
        end_time = int(f.readline().strip())
    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    frame_rate_fraction = Fraction(frame_rate).limit_denominator()
    img_timing_array = []
    for ii in range(frame_num):
        frame_ticks = (
            ii * frame_rate_fraction.denominator * 10**7
        ) // frame_rate_fraction.numerator
        img_timing_array.append(start_time + frame_ticks)
    return np.array(img_timing_array)


def align_to_ref_times(data, data_times, ref_times):
    """Align data to reference timestamps using nearest neighbor."""
    if len(data) == 0:
        # fallback
        return np.zeros((len(ref_times), *data.shape[1:]), dtype=data.dtype)
    nn = NearestNeighbors(n_neighbors=1).fit(data_times.reshape(-1, 1))
    _, indices = nn.kneighbors(ref_times.reshape(-1, 1))
    return data[indices[:, 0]]


def align_files_to_ref_times(files, file_times, ref_times):
    """Align files to reference timestamps using nearest neighbor."""
    nn = NearestNeighbors(n_neighbors=1).fit(file_times.reshape(-1, 1))
    _, indices = nn.kneighbors(ref_times.reshape(-1, 1))
    return [files[i] for i in indices[:, 0]]


# --- Camera parameter utilities ---
def parse_fisheye_intrinsics(params):
    """
    Parse fisheye camera intrinsics.
    params: list or 1D array of length ≥14
    [fx,0,cx,0,fy,cy,0,0,1, k1,k2,k3,k4, ...]
    """
    fx, _, cx, _, fy, cy, *_, k1, k2, k3, k4 = params[:14]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    D = np.array([k1, k2, k3, k4], dtype=np.float64)
    return K, D


def parse_pinhole_intrinsics(K):
    """
    Parse pinhole camera intrinsics.
    params: list or 1D array of length ≥9
    [fx,0,cx,0,fy,cy,0,0,1, ...]
    """
    D = np.zeros((4,), dtype=np.float64)
    return K, D


def quaternion_to_rotation_matrix(q):
    """
    Convert quaternion to rotation matrix.
    q = [qx, qy, qz, qw]
    """
    qx, qy, qz, qw = q
    # Proper normalization using sqrt
    norm = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm < 1e-8:
        return np.eye(3, dtype=np.float64)
    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm

    # Rotation matrix formula
    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz

    R = np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ],
        dtype=np.float64,
    )
    return R


def parse_extrinsic_line(line):
    """
    Parse extrinsic line to camera-to-world transformation matrix.
    line example:
    0.0 637945500675425361  qx   qy   qz   qw   tx      ty      tz   0 0 0 1
    """
    parts = line.strip().split()
    qx, qy, qz, qw = map(float, parts[2:6])
    tx, ty, tz = map(float, parts[6:9])

    R = quaternion_to_rotation_matrix([qx, qy, qz, qw])
    H = np.eye(4, dtype=np.float64)
    H[:3, :3] = R
    H[:3, 3] = [tx, ty, tz]
    return H


# --- Depth processing utilities ---
def depth2disparity(depth):
    """Convert depth to disparity."""
    # This function is not defined in the original code
    # Placeholder implementation - replace with actual implementation
    return 1.0 / (depth + 1e-6)


def align_depth_to_image(
    depth,  # (H_d, W_d) np.ndarray
    depth_K,  # 3×3 fisheye intrinsics
    depth_D,  # (4,) fisheye distortion coeffs
    depth_H,  # 4×4 depth_cam → world
    img_K,  # 3×3 RGB intrinsics
    img_D,  # (k,) RGB distortion coeffs (usually zeros)
    img_H,  # 4×4 RGB_cam → world
    img_size,  # (H_i, W_i)
):
    H_d, W_d = depth.shape
    # 1) Generate all depth pixel coordinates
    u, v = np.meshgrid(np.arange(W_d), np.arange(H_d))
    pts = np.stack([u.ravel(), v.ravel()], axis=-1).astype(np.float32)  # (N,2)

    # 2) Undistort fisheye → normalized plane
    undist = cv2.fisheye.undistortPoints(
        pts.reshape(-1, 1, 2), depth_K, depth_D
    ).reshape(
        -1, 2
    )  # (N,2): [x_norm, y_norm]

    # 3) Back-project to depth camera coords
    z = depth.ravel().astype(np.float32)
    x = undist[:, 0] * z
    y = undist[:, 1] * z
    ones = np.ones_like(z)
    pts_cam = np.vstack([x, y, z, ones])  # (4, N)

    # 4) depth_cam → world
    pts_w = depth_H @ pts_cam

    # 5) world → RGB_cam
    pts_i = np.linalg.inv(img_H) @ pts_w
    pts_i = pts_cam  # Override with original camera points
    pts_i = pts_i[:3, :]  # (3, N)

    # 6) Project using intrinsics
    uvw = img_K @ pts_i  # (3, N)
    u_i = uvw[0, :] / uvw[2, :]
    v_i = uvw[1, :] / uvw[2, :]

    # 7) Integer pixel indices
    u_pix = np.round(u_i).astype(int)
    v_pix = np.round(v_i).astype(int)

    H_i, W_i = img_size
    valid = (u_pix >= 0) & (u_pix < W_i) & (v_pix >= 0) & (v_pix < H_i)

    # 8) Create output map
    aligned = np.zeros((H_i, W_i), dtype=depth.dtype)
    aligned[v_pix[valid], u_pix[valid]] = z[valid]

    return aligned


def process_depth_aligned(
    depth_imgs,  # List[np.ndarray], original depth maps
    depth_intrinsics,  # List[float], fisheye params
    rgb_intrinsics,  # 3×3 K list or np.ndarray
    depth_pose_lines,  # List[str], depth extrinsic lines
    rgb_pose_matrices,  # List[np.ndarray], RGB 4×4 extrinsics
    out_H,
    out_W,  # int, int, output resolution
):
    # 1) Parse intrinsics
    depth_K, depth_D = parse_fisheye_intrinsics(depth_intrinsics)
    rgb_K, rgb_D = rgb_intrinsics, np.zeros((4,), dtype=np.float64)

    # 2) Parse depth extrinsics
    depth_H_list = [parse_extrinsic_line(l) for l in depth_pose_lines]

    # 3) Align each frame → store in list
    aligned_list = []
    for d_img, d_H, r_H in zip(depth_imgs, depth_H_list, rgb_pose_matrices):
        aligned = align_depth_to_image(
            d_img, depth_K, depth_D, d_H, rgb_K, rgb_D, r_H, (out_H, out_W)
        )
        aligned_list.append(aligned)  # shape (H, W)

    # 4) Convert to np.array: shape (N, H, W)
    aligned_depths = np.stack(aligned_list, axis=0)

    return aligned_depths


def line_hand_pose_to_heatmap(
    hand_xyz,  # (T, J, 3)
    cam_poses,  # (T, 4, 4)
    K,  # (3, 3)
    H,
    W,
    sigma=6,
    use_camera_space_deltas=True,
    use_end_joints=False,
):
    T, J, _ = hand_xyz.shape
    hm = np.zeros((T, 3, H, W), np.float32)
    xv = np.arange(W)
    yv = np.arange(H)

    # Calculate joints in camera space (precompute)
    cam_joints = np.zeros((T, J, 3), dtype=np.float32)
    for t in range(T):
        joints_3d = hand_xyz[t]  # (J,3), world coords
        Pw = np.concatenate([joints_3d, np.ones((J, 1))], axis=1).T  # (4,J)
        Pc = np.linalg.inv(cam_poses[t]) @ Pw  # (4,J)
        Pc = axis_transform @ Pc  # (4,J)
        cam_joints[t] = Pc[:3].T  # (J,3)

    # Project all joints to image coordinates
    joint_uvs = np.zeros((T, J, 2), dtype=np.float32)
    for t in range(T):
        Pc = cam_joints[t]  # (J,3)
        uvw = (K @ Pc.T).T  # (J,3)
        joint_uvs[t] = uvw[:, :2] / uvw[:, 2:3]  # (J,2)

    # Calculate deltas (world or camera space)
    if use_camera_space_deltas:
        # Camera space deltas
        deltas = np.zeros_like(cam_joints)
        deltas[:-1] = (cam_joints[1:] - cam_joints[:-1]) * 100  # m -> cm
    else:
        # World space deltas (original method)
        deltas = np.zeros_like(hand_xyz)
        deltas[:-1] = (hand_xyz[1:] - hand_xyz[:-1]) * 100  # m -> cm

    # Limit values with tanh
    colors = np.tanh(deltas)  # Use as color/strength

    # Filter joints if selection is provided
    joints_to_use = range(J)  # Default: use all joints
    if use_end_joints:
        joints_to_use = [0, 5, 9, 13, 17, 21]  # Use only wrist and fingertips

    # For each time step, draw the line to next position and next position
    for t in range(T - 1):  # Skip last frame as it has no next position
        for j in joints_to_use:
            # Get current and next positions
            x1, y1 = joint_uvs[t, j]
            x2, y2 = joint_uvs[t + 1, j]

            # Skip if points are outside image bounds
            if (
                x1 < 0
                or x1 >= W
                or y1 < 0
                or y1 >= H
                or x2 < 0
                or x2 >= W
                or y2 < 0
                or y2 >= H
            ):
                continue

            # Get color from movement delta
            strength = colors[t, j]  # (3,)

            # Draw line between current and next position
            # Create line by interpolating points
            num_steps = max(int(np.hypot(x2 - x1, y2 - y1)), 10)
            for i in range(num_steps):
                alpha = i / num_steps
                x = x1 + alpha * (x2 - x1)
                y = y1 + alpha * (y2 - y1)

                # Draw a small dot for the line
                small_sigma = 1.0
                gx = np.exp(-((xv - x) ** 2) / (2 * small_sigma**2))
                gy = np.exp(-((yv - y) ** 2) / (2 * small_sigma**2))

                for c in range(3):
                    hm[t, c] += (
                        np.outer(gy, gx) * strength[c] * 0.3
                    )  # dim line with delta color

            # Draw next position with a Gaussian
            gx = np.exp(-((xv - x2) ** 2) / (2 * sigma**2))
            gy = np.exp(-((yv - y2) ** 2) / (2 * sigma**2))

            for c in range(3):
                hm[t, c] += (
                    np.outer(gy, gx) * strength[c]
                )  # bright point with delta color

    return hm


# --- Hand trajectory heatmap utilities ---
def hand_pose_to_heatmap(
    hand_xyz,  # (T, J, 3)
    cam_poses,  # (T, 4, 4)
    K,  # (3, 3)
    H,
    W,
    sigma=6,
    use_camera_space_deltas=True,
    use_end_joints=False,  # New parameter: list of joint indices to use (e.g., fingertips and wrist)
):
    T, J, _ = hand_xyz.shape
    hm = np.zeros((T, 3, H, W), np.float32)
    xv = np.arange(W)
    yv = np.arange(H)

    # Calculate joints in camera space (precompute)
    cam_joints = np.zeros((T, J, 3), dtype=np.float32)
    for t in range(T):
        joints_3d = hand_xyz[t]  # (J,3), world coords
        Pw = np.concatenate([joints_3d, np.ones((J, 1))], axis=1).T  # (4,J)
        Pc = np.linalg.inv(cam_poses[t]) @ Pw  # (4,J)
        Pc = axis_transform @ Pc  # (4,J)
        cam_joints[t] = Pc[:3].T  # (J,3)

    # Calculate deltas (world or camera space)
    if use_camera_space_deltas:
        # Camera space deltas
        deltas = np.zeros_like(cam_joints)
        deltas[:-1] = (cam_joints[1:] - cam_joints[:-1]) * 100  # m -> cm
    else:
        # World space deltas (original method)
        deltas = np.zeros_like(hand_xyz)
        deltas[:-1] = (hand_xyz[1:] - hand_xyz[:-1]) * 100  # m -> cm

    # Limit values with tanh
    deltas = np.tanh(deltas)

    # Filter joints if selection is provided
    joints_to_use = range(J)  # Default: use all joints
    if use_end_joints:
        joints_to_use = [0, 5, 9, 13, 17, 21]  # Use only selected joints

    # cam_joints save future cam_joints
    future_cam_joints = np.zeros_like(cam_joints)
    future_cam_joints[:-1] = cam_joints[1:]  # Shift all except last frame
    future_cam_joints[-1] = cam_joints[-1]  # Last frame has no future, use itself
    for t in range(T):
        # Use precomputed camera space joints
        Pc = future_cam_joints[t]  # (J,3)
        uvw = (K @ Pc.T).T
        uv = uvw[:, :2] / uvw[:, 2:3]
        for j in joints_to_use:
            x, y = uv[j]
            gx = np.exp(-((xv - x) ** 2) / (2 * sigma**2))
            gy = np.exp(-((yv - y) ** 2) / (2 * sigma**2))
            strength = deltas[t, j]  # Use delta magnitude as strength
            for c in range(3):
                hm[t, c] += np.outer(gy, gx) * strength[c]
    return hm


def generate_raymap(
    headposes: np.ndarray, rgb_info: np.ndarray, H: int = 256, W: int = 454
) -> np.ndarray:
    T = headposes.shape[0]
    E = headposes.reshape(T, 4, 4)
    Ks = rgb_info[:, 16:25].reshape(T, 3, 3)

    rm_list = []
    for i in range(T):
        K = Ks[i]
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        # Create meshgrid
        x, y = np.meshgrid(np.arange(W), np.arange(H), indexing="xy")

        # Calculate camera dirs (similar to reference)
        camera_dirs = np.stack(
            [(x - cx + 0.5) / fx, (y - cy + 0.5) / fy, np.ones_like(x)], axis=-1
        )  # [H, W, 3]

        # Calculate rays_d
        rays_d = camera_dirs @ E[i, :3, :3].T  # [H, W, 3]

        # Get rays_o (camera origin in world space)
        t = E[i, :3, 3]
        rays_o = np.broadcast_to(t, (H, W, 3))

        # Apply log transformation to rays_o as in the original code
        rays_o_log = np.sign(rays_o) * np.log1p(np.abs(rays_o))

        # Combine and transpose to get raymap
        rm = np.concatenate([rays_o_log, rays_d], axis=-1).transpose(2, 0, 1)
        rm_list.append(rm)

    rm = np.stack(rm_list)
    return rm


def generate_delta_raymap(
    headposes: np.ndarray, rgb_info: np.ndarray, H: int = 256, W: int = 454
) -> np.ndarray:
    """
    Generate raymap with delta information between current and future head poses.

    Args:
        headposes: (T, 16) array or (T, 4, 4) array of head poses
        rgb_info: (T, K) array containing camera intrinsics
        H, W: Output height and width

    Returns:
        raymap: (T, 12, H, W) array containing:
            - channels 0-2: ray origins (log-transformed)
            - channels 3-5: ray directions
            - channels 6-8: translation delta (log-transformed)
            - channels 9-11: rotation delta
    """
    T = headposes.shape[0]
    E = headposes.reshape(T, 4, 4)
    Ks = rgb_info[:, 16:25].reshape(T, 3, 3)

    # Calculate deltas (skip last frame)
    deltas_t = np.zeros((T, 3), dtype=np.float32)  # translation deltas
    deltas_r = np.zeros((T, 3, 3), dtype=np.float32)  # rotation deltas

    # Compute deltas between consecutive frames
    for i in range(T - 1):
        # Translation delta (current to next)
        t_curr = E[i, :3, 3]
        t_next = E[i + 1, :3, 3]
        deltas_t[i] = (t_next - t_curr) * 100  # Convert to cm

        # Rotation delta (current to next)
        R_curr = E[i, :3, :3]
        R_next = E[i + 1, :3, :3]
        # R_delta represents the rotation from current to next frame
        R_delta = R_next @ R_curr.T
        # Convert to a more compact representation
        deltas_r[i] = R_delta - np.eye(
            3
        )  # Simple difference to represent rotation change

    # Apply log transformation to translation deltas (same as rays_o)
    deltas_t_log = np.sign(deltas_t) * np.log1p(np.abs(deltas_t))

    rm_list = []
    for i in range(T):
        K = Ks[i]
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        # Create meshgrid
        x, y = np.meshgrid(np.arange(W), np.arange(H), indexing="xy")

        # Calculate camera dirs
        camera_dirs = np.stack(
            [(x - cx + 0.5) / fx, (y - cy + 0.5) / fy, np.ones_like(x)], axis=-1
        )  # [H, W, 3]

        # Calculate rays_d
        rays_d = camera_dirs @ E[i, :3, :3].T  # [H, W, 3]

        # Get rays_o (camera origin in world space)
        t = E[i, :3, 3]
        rays_o = np.broadcast_to(t, (H, W, 3))

        # Apply log transformation to rays_o
        rays_o_log = np.sign(rays_o) * np.log1p(np.abs(rays_o))

        # Create delta information visualization
        # Broadcast delta movement to full image size
        delta_t = np.broadcast_to(deltas_t_log[i], (H, W, 3))

        # Create rotational movement visualization
        if i < T - 1:
            delta_dir = rays_d @ deltas_r[i].T  # [H, W, 3]
        else:
            delta_dir = np.zeros((H, W, 3), dtype=np.float32)

        # Combine and transpose to get raymap with deltas (keeping translation and rotation separate)
        rm = np.concatenate(
            [rays_o_log, rays_d, delta_t, delta_dir], axis=-1
        ).transpose(2, 0, 1)
        rm_list.append(rm)

    rm = np.stack(rm_list)
    return rm


# ---------- Dataset Core ----------
class MultiRawDataset(VisionDataset):
    def __init__(
        self,
        root,
        label_root,
        subset,
        max_samples=16,
        transform=None,
        target_transform=None,
    ):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.root = root
        self.label_root = label_root
        self.subset = subset
        self.max_samples = max_samples
        self.sessions = self._load_sessions()

    def _load_sessions(self):
        session_list_path = os.path.join(self.label_root, f"{self.subset}_0724.txt")
        with open(session_list_path) as f:
            return [line.strip() for line in f if line.strip()]

    def __getitem__(self, idx):
        fps = 30  # random.choice([8, 10, 12, 15, 24])
        # fps = None
        session = self.sessions[idx]
        base_path = os.path.join(self.root, session, "Export_py")
        pose_arr = read_pose_txt(os.path.join(base_path, "Video", "Pose_sync.txt"))
        ref_times_full = pose_arr[:, 1]
        num_frames = len(ref_times_full)

        # FPS에 따른 프레임 샘플링
        if fps is not None:
            # HoLi 데이터셋 원본은 30fps로 가정
            original_fps = 30
            # 원하는 fps에 따른 프레임 간격 계산
            frame_interval = original_fps / fps

            # 샘플링할 총 기간 계산
            sample_duration_frames = (self.max_samples - 1) * frame_interval

            if sample_duration_frames >= num_frames:
                ref_idxs = np.arange(min(num_frames, self.max_samples))
            else:
                max_start_idx = int(num_frames - sample_duration_frames - 1)
                _random_init = np.random.randint(0, max(1, max_start_idx))

                ref_idxs = np.array(
                    [
                        _random_init + int(i * frame_interval)
                        for i in range(self.max_samples)
                    ]
                )

                ref_idxs = ref_idxs[ref_idxs < num_frames]
        else:
            if num_frames > self.max_samples:
                _random_init = 0  # np.random.randint(0, num_frames - self.max_samples)
                ref_idxs = np.linspace(
                    _random_init, _random_init + self.max_samples - 1, self.max_samples
                ).astype(int)
            else:
                ref_idxs = np.arange(num_frames)

        ref_times = ref_times_full[ref_idxs]

        # Get video frame times
        video_path = os.path.join(base_path, "Video_pitchshift.mp4")
        timing_path = os.path.join(base_path, "Video", "VideoMp4Timing.txt")
        img_timing_array = get_video_frame_times(video_path, timing_path)

        # Find nearest frames
        search = IndexSearch(img_timing_array)
        rgb_frame_indices = [search.nearest_neighbor(ts) for ts in ref_times]

        # Read RGB frames
        cap = cv2.VideoCapture(video_path)
        rgb_frames = []
        for fidx in rgb_frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fidx))
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError(f"Frame read failed at index {fidx}")
            rgb_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()

        # Stack and rearrange
        rgb = np.stack(rgb_frames)  # (T, H, W, C)
        rgb = einops.rearrange(rgb, "t h w c -> t c h w")  # (T, C, H, W)
        H, W = rgb.shape[2:]

        head = pose_arr[:, 2:18].reshape(-1, 4, 4)[ref_idxs]

        # Read camera intrinsics
        Kr = read_intrinsics_txt(os.path.join(base_path, "Video", "Intrinsics.txt"))
        rgb_info = np.hstack(
            [head.reshape(len(ref_idxs), 16), np.tile(Kr.flatten(), (len(ref_idxs), 1))]
        )

        # Read and align hand poses
        left_hand = read_hand_pose_txt(
            os.path.join(base_path, "Hands", "Left_sync.txt")
        )
        left_hand_times = left_hand[:, 1]
        left_hand_aligned = align_to_ref_times(left_hand, left_hand_times, ref_times)

        right_hand = read_hand_pose_txt(
            os.path.join(base_path, "Hands", "Right_sync.txt")
        )
        right_hand_times = right_hand[:, 1]
        right_hand_aligned = align_to_ref_times(right_hand, right_hand_times, ref_times)

        # ---- Read depth frames ----
        depth_video_path = os.path.join(base_path, "Depth_compress.mp4")
        cap = cv2.VideoCapture(depth_video_path)
        depth_frames = []
        for fidx in rgb_frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fidx))
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError(f"Frame read failed at index {fidx}")
            depth_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        cap.release()

        depth_frames = np.stack(depth_frames)
        # ---- Generate hand trajectory heatmaps ----
        hand_l_xyz = left_hand_aligned[:, 4:].reshape(len(ref_idxs), -1, 3)
        hand_r_xyz = right_hand_aligned[:, 4:].reshape(len(ref_idxs), -1, 3)
        hand_l_hm = hand_pose_to_heatmap(hand_l_xyz, head, Kr, H, W, sigma=4)
        hand_r_hm = hand_pose_to_heatmap(hand_r_xyz, head, Kr, H, W, sigma=4)

        # ---- Generate raymap ----
        raymap = generate_raymap(head, rgb_info, H, W)

        # ---- Padding function (for when num_frames < max_samples) ----
        def pad(arr):
            arr = np.array(arr)
            if arr.shape[0] < self.max_samples:
                pads = [(0, self.max_samples - arr.shape[0])] + [(0, 0)] * (
                    arr.ndim - 1
                )
                arr = np.pad(arr, pads, mode="constant")
            return arr

        # Prepare sample dictionary
        sample = {
            "rgb": pad(rgb),
            "head": pad(head),
            "hands-left": pad(hand_l_xyz),
            "hands-right": pad(hand_r_xyz),
            "depth": pad(depth_frames),
            "hands-left-traj": pad(hand_l_hm),
            "hands-right-traj": pad(hand_r_hm),
            "raymap": pad(raymap),
            "fps": fps,
        }

        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.sessions)
