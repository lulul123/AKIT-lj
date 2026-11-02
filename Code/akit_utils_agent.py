#!/usr/bin/env python3
"""A-KIT 数据读取、时间对齐、Dataset 与可微 Kalman 公共实现。

该模块把一条轨迹的 IMU、速度、Phi、PStart、H、R、Inno、Qc、名义状态和参考位置
对齐成训练样本，并实现模型输出之后的逐步协方差传播与位置更新。训练入口和 smoke
test 共用这里的科学计算逻辑。它不定义神经网络，也不管理实验配置和 checkpoint。
"""

import os
import math
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import Dataset


def sample_window_key(timestamp):
    """Return k for the left-closed, right-open sample window [k-1, k)."""
    return math.floor(timestamp) + 1


def phi_window_key(timestamp, tolerance=1e-7):
    """Return k for a Phi propagation endpoint in the right-closed window (k-1, k]."""
    return math.ceil(float(timestamp) - tolerance)


def load_truePos(file_path):
    data_dict = {}
    try:
        with open(file_path, "r") as f:
            line_count = 0
            header_found = False
            for line in f:
                line_count += 1
                line = line.strip()
                if not line or line.startswith('#') or 'Profile:' in line or 'Source:' in line:
                    continue
                if 'Week' in line and 'GPSTime' in line and not header_found:
                    header_found = True
                    continue
                if 'week' in line and '(sec)' in line:
                    continue
                if not header_found:
                    continue
                values = line.split()
                if len(values) < 15:
                    continue
                try:
                    week = int(values[0])
                    gps_time = float(values[1])
                    time_stamp = gps_time
                    if len(values) >= 13:
                        x_ll = float(values[9])
                        y_ll = float(values[10])
                        z_ll = float(values[11])
                        pos = np.array([x_ll, y_ll, z_ll])
                        data_dict[time_stamp] = pos
                except (ValueError, IndexError):
                    continue
            print(f"  成功读取 {len(data_dict)} 条位置数据 (总共{line_count}行)")
            if data_dict:
                times = list(data_dict.keys())
                print(f"  时间范围: {min(times)} 到 {max(times)}")
    except Exception as e:
        print(f"  读取位置文件出错: {e}")
    return data_dict


def load_matrix_file(file_path, rows, cols):
    data_dict = {}
    try:
        with open(file_path, 'r') as f:
            header = f.readline().strip()
            lines = f.readlines()
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                if not line:
                    i += 1
                    continue
                values = line.split()
                try:
                    time_stamp = int(values[0])
                    if len(values) > 1:
                        matrix_data = []
                        first_row = [float(v) for v in values[1:]]
                        if len(first_row) >= cols:
                            matrix_data.extend(first_row[:cols])
                        else:
                            matrix_data.extend(first_row + [0.0] * (cols - len(first_row)))
                        for row_idx in range(1, rows):
                            i += 1
                            if i < len(lines):
                                row_line = lines[i].strip()
                                if row_line:
                                    row_values = [float(v) for v in row_line.split()]
                                    if len(row_values) >= cols:
                                        matrix_data.extend(row_values[:cols])
                                    else:
                                        matrix_data.extend(row_values + [0.0] * (cols - len(row_values)))
                                else:
                                    matrix_data.extend([0.0] * cols)
                            else:
                                matrix_data.extend([0.0] * cols)
                        if len(matrix_data) >= rows * cols:
                            matrix = np.array(matrix_data[:rows * cols]).reshape(rows, cols)
                            data_dict[time_stamp] = matrix
                    else:
                        matrix_data = []
                        for row_idx in range(rows):
                            i += 1
                            if i < len(lines):
                                row_line = lines[i].strip()
                                if row_line:
                                    row_values = [float(v) for v in row_line.split()]
                                    if len(row_values) >= cols:
                                        matrix_data.extend(row_values[:cols])
                                    else:
                                        matrix_data.extend(row_values + [0.0] * (cols - len(row_values)))
                                else:
                                    matrix_data.extend([0.0] * cols)
                            else:
                                matrix_data.extend([0.0] * cols)
                        if len(matrix_data) >= rows * cols:
                            matrix = np.array(matrix_data[:rows * cols]).reshape(rows, cols)
                            data_dict[time_stamp] = matrix
                except ValueError:
                    pass
                i += 1
            print(f"  成功读取 {len(data_dict)} 条矩阵数据")
    except Exception as e:
        print(f"  读取文件出错: {e}")
    return data_dict


def load_timed_matrix_sequence(file_path, rows, cols):
    """Load all timestamped matrices without collapsing sub-second samples."""
    records = []
    with open(file_path, "r") as f:
        f.readline()
        lines = f.readlines()
    i = 0
    while i < len(lines):
        values = lines[i].strip().split()
        if len(values) < 2:
            i += 1
            continue
        try:
            timestamp = float(values[0])
            dt = float(values[1])
        except ValueError:
            i += 1
            continue
        matrix_values = [float(v) for v in values[2:]]
        while len(matrix_values) < rows * cols and i + 1 < len(lines):
            i += 1
            matrix_values.extend(float(v) for v in lines[i].strip().split())
        if len(matrix_values) >= rows * cols:
            records.append((timestamp, dt, np.asarray(matrix_values[:rows * cols], dtype=np.float64).reshape(rows, cols)))
        i += 1
    records.sort(key=lambda item: item[0])
    return records


def load_constant_matrix(file_path, rows, cols):
    """Load a matrix file containing one header and one matrix without a timestamp."""
    with open(file_path, "r") as f:
        f.readline()
        values = [float(v) for line in f for v in line.strip().split()]
    if len(values) < rows * cols:
        raise ValueError(f"{file_path} contains {len(values)} values; expected {rows * cols}")
    return np.asarray(values[:rows * cols], dtype=np.float64).reshape(rows, cols)


def load_gnss_mech(file_path):
    """Load GNSS-update prior nominal states: pos, vel, attitude, accel bias, gyro bias."""
    result = {}
    with open(file_path, "r") as f:
        f.readline()
        for line in f:
            values = line.strip().split()
            if len(values) < 16:
                continue
            timestamp = float(values[0])
            result[int(timestamp)] = np.asarray([float(v) for v in values[1:16]], dtype=np.float64).reshape(15, 1)
    return result


def check_time_alignment(data_dir):
    required_files = ["imu.txt", "Vel.txt", "Inno.txt", "Phi.txt", "H.txt", "R.txt", "PStart.txt", "GNSSMech.txt", "Qc.txt"]
    for file in required_files:
        file_path = os.path.join(data_dir, file)
        if not os.path.exists(file_path):
            print(f"Missing file: {file}")
            return False
    try:
        dataset = MultiDatasetIMU([data_dir], seq_len=100)
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"✓ Data alignment successful")
            print(f"  - Available samples: {len(dataset)}")
            print(f"  - Sample high-freq time: {sample['time']}")
            print(f"  - Sample matrix time: {sample['matrix_time']}")
            return True
        else:
            print(f"✗ No aligned samples found")
            return False
    except Exception as e:
        print(f"✗ Error during alignment: {e}")
        return False


class MultiDatasetIMU(Dataset):
    def __init__(self, data_dirs, seq_len=100):
        self.seq_len = seq_len
        self.all_samples = []
        for data_dir in data_dirs:
            print(f"Loading dataset from: {data_dir}")
            samples = self._load_single_dataset(data_dir)
            self.all_samples.extend(samples)
            print(f"Added {len(samples)} samples from {data_dir}")
        self.max_phi_steps = max((len(sample["Phi_dt"]) for sample in self.all_samples), default=0)
        print(f"Total samples across all datasets: {len(self.all_samples)}")

    def _load_single_dataset(self, data_dir):
        imu_file = os.path.join(data_dir, "imu.txt")
        vel_file = os.path.join(data_dir, "Vel.txt")
        pos_files = [f for f in os.listdir(data_dir) if f.lower().endswith((".imr.ref", ".pos", ".ref"))]
        if not pos_files:
            raise FileNotFoundError(f"No position reference file found in {data_dir}")
        pos_file = os.path.join(data_dir, pos_files[0])
        matrix_files = {
            "phi": os.path.join(data_dir, "Phi.txt"),
            "H": os.path.join(data_dir, "H.txt"),
            "R": os.path.join(data_dir, "R.txt"),
            "P_start": os.path.join(data_dir, "PStart.txt"),
            "state": os.path.join(data_dir, "GNSSMech.txt"),
            "inno": os.path.join(data_dir, "Inno.txt"),
            "qc": os.path.join(data_dir, "Qc.txt")
        }
        imu_second_data, imu_timestamps = self._load_and_group_imu_data(imu_file, 6)
        vel_second_data = self._load_vel_data_by_imu_timestamps(vel_file, imu_timestamps, 3)
        common_seconds = set(imu_second_data.keys()) & set(vel_second_data.keys())
        print(f"Found {len(common_seconds)} seconds with complete 100Hz data")
        pos_dict = load_truePos(pos_file)
        phi_records = load_timed_matrix_sequence(matrix_files["phi"], 15, 15) if os.path.exists(matrix_files["phi"]) else []
        phi_by_second = defaultdict(list)
        for phi_time, phi_dt, phi_matrix in phi_records:
            phi_by_second[phi_window_key(phi_time)].append((phi_time, phi_dt, phi_matrix))
        H_dict = load_matrix_file(matrix_files["H"], 3, 15) if os.path.exists(matrix_files["H"]) else {}
        R_dict = load_matrix_file(matrix_files["R"], 3, 3) if os.path.exists(matrix_files["R"]) else {}
        P_dict = load_matrix_file(matrix_files["P_start"], 15, 15) if os.path.exists(matrix_files["P_start"]) else {}
        state_dict = load_gnss_mech(matrix_files["state"]) if os.path.exists(matrix_files["state"]) else {}
        inno_dict = load_matrix_file(matrix_files["inno"], 3, 1) if os.path.exists(matrix_files["inno"]) else {}
        Qc = load_constant_matrix(matrix_files["qc"], 15, 15) if os.path.exists(matrix_files["qc"]) else None
        print(f"Matrix data loaded: pos({len(pos_dict)}), Phi_steps({len(phi_records)}), H({len(H_dict)}), R({len(R_dict)}), P_start({len(P_dict)}), state_pred({len(state_dict)}), inno({len(inno_dict)}), Qc({'yes' if Qc is not None else 'no'})")

        # --- 新增: 基线对齐（开始） ---
        # 目标：使用 pos_true 与 stateX 的首个对齐时刻作为 baseline，优先精确时间匹配，
        # 否则对 pos 时间按 nearest-within-threshold 匹配 state 时间。
        baseline_pos = None
        baseline_time = None
        try:
            pos_times = sorted(list(pos_dict.keys()))
            state_times = sorted(list(state_dict.keys()))
            if pos_times and state_times:
                # 优先寻找精确相同的时间点
                common_times = set(pos_times).intersection(state_times)
                if common_times:
                    baseline_time = min(common_times)
                    baseline_pos = np.array(pos_dict[baseline_time], dtype=np.float64)
                else:
                    # 最近邻匹配，阈值可调整
                    tol = 0.5  # seconds
                    found = False
                    for pt in pos_times:
                        # 找到与 pt 最近的 state 时间
                        closest = min(state_times, key=lambda st: abs(st - pt))
                        if abs(closest - pt) <= tol:
                            baseline_time = pt
                            baseline_pos = np.array(pos_dict[pt], dtype=np.float64)
                            found = True
                            break
                    if not found:
                        # 回退：使用 pos 的最早时间作为 baseline（保证有基线）
                        baseline_time = pos_times[0]
                        baseline_pos = np.array(pos_dict[baseline_time], dtype=np.float64)
                # 打印基线信息，便于调试与回退
                # print(f"  Baseline selected at time={baseline_time}, baseline_pos={baseline_pos}")
        except Exception as e:
            print(f"  Baseline selection failed: {e}")
            baseline_pos = None
            baseline_time = None
        # --- 新增: 基线对齐（结束） ---

        matrix_times = set(pos_dict.keys()) & set(phi_by_second.keys()) & set(H_dict.keys()) & \
                       set(R_dict.keys()) & set(P_dict.keys()) & set(state_dict.keys()) & \
                       set(inno_dict.keys())
        valid_seconds = common_seconds & matrix_times
        print(f"Found {len(valid_seconds)} seconds with aligned data")
        samples = []
        successful_gnss_times = sorted(state_dict)
        previous_success = {
            current: previous for previous, current in zip(successful_gnss_times, successful_gnss_times[1:])
        }
        skipped_nonunit_intervals = 0
        skipped_invalid_phi_intervals = 0
        for second in sorted(valid_seconds):
            # PStart(second) begins at the previous successful GNSS boundary. The
            # current network input covers exactly one second, so reject outage
            # intervals instead of replaying only their final second of Phi.
            if previous_success.get(second) != second - 1:
                skipped_nonunit_intervals += 1
                continue
            phi_steps = sorted(phi_by_second[second], key=lambda item: item[0])
            if not phi_steps:
                continue
            phi_tolerance = 5e-6
            starts_at_boundary = abs((phi_steps[0][0] - phi_steps[0][1]) - (second - 1)) <= phi_tolerance
            ends_at_boundary = abs(phi_steps[-1][0] - second) <= phi_tolerance
            positive_dt = all(item[1] > 0.0 for item in phi_steps)
            continuous = all(
                abs((current[0] - current[1]) - previous[0]) <= phi_tolerance
                for previous, current in zip(phi_steps, phi_steps[1:])
            )
            if not (starts_at_boundary and ends_at_boundary and positive_dt and continuous):
                skipped_invalid_phi_intervals += 1
                continue
            imu_data_100 = imu_second_data[second]
            vel_data_100 = vel_second_data[second]
            combined_data = []
            for i in range(self.seq_len):
                combined = imu_data_100[i] + vel_data_100[i]
                combined_data.append(combined)
            # 取得位置与 state，并转换为相对基线（若存在基线）
            pos_abs = pos_dict[second]
            # 不对 pos 做任何基线或额外处理，直接使用读取值以便只做时间匹配
            pos_rel = np.array(pos_abs, dtype=np.float64)

            stateX_mat = state_dict[second]
            try:
                # 仅转换为 numpy 数组，禁用对 state 的基线减法或其它修改
                stateX_mat = np.array(stateX_mat, dtype=np.float64)
            except Exception:
                pass

            sample = {
                "imu": np.array(combined_data, dtype=np.float64),
                "time": float(second),
                "matrix_time": second,
                "pos_true": pos_rel,
                "Phi_seq": np.stack([item[2] for item in phi_steps], axis=0),
                "Phi_dt": np.asarray([item[1] for item in phi_steps], dtype=np.float64),
                "H": H_dict[second],
                "R": R_dict[second],
                "P_start": P_dict[second],
                "state_pred": stateX_mat,
                "Inno": inno_dict[second],
                "Qc": Qc,
            }
            samples.append(sample)
        if skipped_nonunit_intervals:
            print(f"Skipped {skipped_nonunit_intervals} samples whose successful GNSS interval was not one second")
        if skipped_invalid_phi_intervals:
            print(f"Skipped {skipped_invalid_phi_intervals} samples whose Phi sequence did not continuously cover (k-1, k]")
        return samples

    def _load_and_group_imu_data(self, imu_file, data_dim):
        second_groups = {}
        second_timestamps = {}
        with open(imu_file, "r") as f:
            header = f.readline()
            for line in f:
                values = line.strip().split()
                if len(values) < data_dim + 1:
                    continue
                full_time = float(values[0])
                second = sample_window_key(full_time)
                data_values = [float(v) for v in values[1:data_dim+1]]
                if second not in second_groups:
                    second_groups[second] = []
                    second_timestamps[second] = []
                second_groups[second].append(data_values)
                second_timestamps[second].append(full_time)
        filtered_groups = {}
        filtered_timestamps = {}
        for second, data_list in second_groups.items():
            if len(data_list) == self.seq_len:
                filtered_groups[second] = data_list
                filtered_timestamps[second] = second_timestamps[second]
            else:
                pass
        return filtered_groups, filtered_timestamps

    def _load_vel_data_by_imu_timestamps(self, vel_file, imu_timestamps, data_dim):
        vel_data_all = {}
        vel_start_second = None
        vel_end_second = None
        with open(vel_file, "r") as f:
            header = f.readline()
            for line in f:
                values = line.strip().split()
                if len(values) < data_dim + 1:
                    continue
                full_time = float(values[0])
                data_values = [float(v) for v in values[1:data_dim+1]]
                vel_data_all[full_time] = data_values
                second = sample_window_key(full_time)
                if vel_start_second is None:
                    vel_start_second = second
                    vel_end_second = second
                else:
                    vel_start_second = min(vel_start_second, second)
                    vel_end_second = max(vel_end_second, second)
        print(f"Total Vel data points loaded: {len(vel_data_all)}")
        vel_data_by_second = {}
        for vel_time, vel_data in vel_data_all.items():
            second = sample_window_key(vel_time)
            if second not in vel_data_by_second:
                vel_data_by_second[second] = {}
            vel_data_by_second[second][vel_time] = vel_data
        filtered_vel_groups = {}
        for second, imu_times in imu_timestamps.items():
            if second < vel_start_second or second > vel_end_second:
                continue
            if second not in vel_data_by_second:
                continue
            vel_data_for_second = []
            matched_count = 0
            second_vel_data = vel_data_by_second[second]
            for imu_time in imu_times:
                best_match = None
                min_diff = float('inf')
                for vel_time in second_vel_data.keys():
                    time_diff = abs(vel_time - imu_time)
                    if time_diff < min_diff:
                        min_diff = time_diff
                        best_match = vel_time
                if best_match is not None and min_diff < 0.002:
                    vel_data_for_second.append(second_vel_data[best_match])
                    matched_count += 1
                else:
                    if vel_data_for_second:
                        vel_data_for_second.append(vel_data_for_second[-1])
                    else:
                        vel_data_for_second.append([0.0] * data_dim)
            if len(vel_data_for_second) == self.seq_len:
                filtered_vel_groups[second] = vel_data_for_second
        return filtered_vel_groups

    def __len__(self):
        return len(self.all_samples)

    def __getitem__(self, idx):
        sample = self.all_samples[idx]
        imu_tensor = torch.tensor(sample["imu"], dtype=torch.float64)
        imu_tensor = imu_tensor.permute(1, 0)
        phi_steps = len(sample["Phi_dt"])
        phi_seq = torch.eye(15, dtype=torch.float64).repeat(self.max_phi_steps, 1, 1)
        phi_dt = torch.zeros(self.max_phi_steps, dtype=torch.float64)
        phi_mask = torch.zeros(self.max_phi_steps, dtype=torch.bool)
        if phi_steps:
            phi_seq[:phi_steps] = torch.tensor(sample["Phi_seq"], dtype=torch.float64)
            phi_dt[:phi_steps] = torch.tensor(sample["Phi_dt"], dtype=torch.float64)
            phi_mask[:phi_steps] = True
        return {
            "imu": imu_tensor,
            "time": torch.tensor(sample["time"], dtype=torch.float64),
            "matrix_time": torch.tensor(sample["matrix_time"], dtype=torch.float64),
            "pos_true": torch.tensor(sample["pos_true"], dtype=torch.float64),
            "Phi_seq": phi_seq,
            "Phi_dt": phi_dt,
            "Phi_mask": phi_mask,
            "H": torch.tensor(sample["H"], dtype=torch.float64),
            "R": torch.tensor(sample["R"], dtype=torch.float64),
            "P_start": torch.tensor(sample["P_start"], dtype=torch.float64),
            "state_pred": torch.tensor(sample["state_pred"], dtype=torch.float64),
            "Inno": torch.tensor(sample["Inno"], dtype=torch.float64),
            "Qc": torch.tensor(sample["Qc"], dtype=torch.float64),
        }


class IMUDataset(Dataset):
    def __init__(self, data_dir, seq_len=100):
        self.multi_dataset = MultiDatasetIMU([data_dir], seq_len)
    def __len__(self):
        return len(self.multi_dataset)
    def __getitem__(self, idx):
        return self.multi_dataset[idx]


def kalman_filter(Phi_seq, Phi_dt, H, R, P_start, Inno, state_pred, q_scale, Qc_origin, Phi_mask=None):
    """Differentiable stepwise covariance prediction and GNSS position update.
    Inputs (batch first):
      Phi_seq: (B, T, n, n) -- per-step transitions, padded with identity
      Phi_dt: (B, T) -- per-step propagation durations, padded with zero
      Phi_mask: optional (B, T) boolean mask for valid propagation steps
      H:   (B, m, n)
      R:   (B, m, m)
      P_start: (B, n, n) -- posterior covariance at interval start
      Inno:(B, m, 1)
      state_pred:(B, n, 1) -- state immediately before the GNSS update
      q_scale:(B, n) -- positive state-wise scales for Qc_origin
      Qc_origin:(B, n, n) -- continuous-time process-noise intensity
    Returns:
      pos_predict: (B, 3) -- first 3 components of updated state
    This implementation uses torch operations only and adds small jitter to S diagonal
    for numerical stability. Uses torch.linalg.solve when possible.
    """
    eps = 1e-8
    device = Phi_seq.device
    dtype = Phi_seq.dtype
    # 强制使用 double 精度进行卡尔曼内部计算，避免 float/double mismatch（保留 autograd）
    try:
        target_dtype = torch.float64
        Phi_seq = Phi_seq.to(device=device, dtype=target_dtype)
        Phi_dt = Phi_dt.to(device=device, dtype=target_dtype)
        H = H.to(device=device, dtype=target_dtype)
        R = R.to(device=device, dtype=target_dtype)
        P_start = P_start.to(device=device, dtype=target_dtype)
        Inno = Inno.to(device=device, dtype=target_dtype)
        if state_pred is not None:
            state_pred = state_pred.to(device=device, dtype=target_dtype)
        q_scale = q_scale.to(device=device, dtype=target_dtype)
        Qc_origin = Qc_origin.to(device=device, dtype=target_dtype)
        if Phi_mask is not None:
            Phi_mask = Phi_mask.to(device=device, dtype=torch.bool)
        dtype = target_dtype
    except Exception:
        pass

    # Build the predicted continuous-time Qc. D @ Qc @ D preserves symmetry
    # and positive semidefiniteness while scaling Qc's diagonal by q_scale.
    try:
        if q_scale.dim() == 2:
            D = torch.diag_embed(torch.sqrt(torch.clamp(q_scale, min=0.0)))
            Qc_pred = torch.matmul(torch.matmul(D, Qc_origin), D)
        elif q_scale.dim() == 3 and q_scale.shape == Qc_origin.shape:
            Qc_pred = Qc_origin * q_scale
        else:
            raise ValueError(f"Unexpected q_scale shape: {tuple(q_scale.shape)}; expected (B,n) or (B,n,n) matching Qc_origin {tuple(Qc_origin.shape)}")
    except Exception as e:
        raise RuntimeError(f"Failed to construct predicted Qc from q_scale: {e}")

    # 支持 H/R 为非批量（广播）
    if H.dim() == 2:
        H = H.unsqueeze(0).expand(P_start.shape[0], -1, -1)
    if R.dim() == 2:
        R = R.unsqueeze(0).expand(P_start.shape[0], -1, -1)

    if Phi_seq.dim() == 3:
        Phi_seq = Phi_seq.unsqueeze(0)
    if Phi_dt.dim() == 1:
        Phi_dt = Phi_dt.unsqueeze(0)
    if Phi_mask is None:
        Phi_mask = Phi_dt > 0.0
    elif Phi_mask.dim() == 1:
        Phi_mask = Phi_mask.unsqueeze(0)
    if Phi_seq.shape[:2] != Phi_dt.shape or Phi_dt.shape != Phi_mask.shape:
        raise ValueError("Phi_seq, Phi_dt and Phi_mask must share batch and step dimensions")

    P_pred = P_start
    for step in range(Phi_seq.shape[1]):
        Phi_i = Phi_seq[:, step]
        dt_i = Phi_dt[:, step].unsqueeze(-1).unsqueeze(-1)
        Phi_i_t = Phi_i.transpose(-2, -1)
        Q_i = 0.5 * (torch.matmul(torch.matmul(Phi_i, Qc_pred), Phi_i_t) + Qc_pred) * dt_i
        P_next = torch.matmul(torch.matmul(Phi_i, P_pred), Phi_i_t) + Q_i
        P_pred = torch.where(Phi_mask[:, step, None, None], P_next, P_pred)

    # 使用先验协方差进行更新
    HP = torch.matmul(H, P_pred)           # (B, m, n)
    HPHt = torch.matmul(HP, H.transpose(-2, -1))  # (B, m, m)
    # 在 S 的对角上加一个极小的 jitter 以提高数值稳定性，避免奇异矩阵
    S = HPHt + R + 1e-12 * torch.eye(HPHt.shape[-1], device=device, dtype=dtype).unsqueeze(0)

    # 计算 S 的逆（batched）
    try:
        S_inv = torch.linalg.inv(S)
    except Exception:
        S_inv = torch.inverse(S)

    K = torch.matmul(torch.matmul(P_pred, H.transpose(-2, -1)), S_inv)  # (B, n, m)

    # In the ECEF loose-coupling convention Inno=INS-GNSS and StateX is the
    # estimated navigation error, so nominal position feedback is subtraction.
    state_update = torch.matmul(K, Inno)  # (B, n, 1)
    position_post = state_pred[:, 0:3, :] - state_update[:, 0:3, :]

    # ====== 合并后的精简 DEBUG 打印 (仅打印第一个样本的前3维) ======
    if state_pred is not None:
        try:
            with torch.no_grad():
                # 提取数据
                s_pre = state_pred[0, :3, 0].cpu().numpy()
                s_upd = state_update[0, :3, 0].cpu().numpy()
                s_pst = position_post[0, :3, 0].cpu().numpy()
                
                # 格式化函数
                fmt = lambda x: "[" + ",".join([f"{v:.4f}" for v in x]) + "]"
                
                # 单行打印核心链路
                print(f"DEBUG KF [Pos]: Pre:{fmt(s_pre)} - Error:{fmt(s_upd)} = Pst:{fmt(s_pst)}")
        except Exception as e:
            print(f"DEBUG KF: Print failed ({e})")
    # ==========================================================

    # Joseph form 保证数值稳定并保持半正定
    I = torch.eye(P_start.shape[-1], device=device, dtype=dtype).unsqueeze(0).expand(P_start.shape[0], -1, -1)
    KH = torch.matmul(K, H)  # (B, n, n)
    temp = I - KH
    P_post = torch.matmul(torch.matmul(temp, P_pred), temp.transpose(-2, -1)) + torch.matmul(torch.matmul(K, R), K.transpose(-2, -1))

    return position_post.squeeze(-1)
