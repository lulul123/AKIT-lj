#!/usr/bin/env python3
# akit_utils.py
# 公共工具：数据读取、时间对齐、数据集与 Kalman 计算

import os
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import Dataset


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


def check_time_alignment(data_dir):
    required_files = ["imu.txt", "Vel.txt", "Inno.txt", "Phi.txt", "H.txt", "R.txt", "StateP.txt", "StateX_new.txt", "Q.txt"]
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
            "P": os.path.join(data_dir, "StateP.txt"),
            "state": os.path.join(data_dir, "StateX_new.txt"),
            "inno": os.path.join(data_dir, "Inno.txt"),
            "Q": os.path.join(data_dir, "Q.txt")
        }
        imu_second_data, imu_timestamps = self._load_and_group_imu_data(imu_file, 6)
        vel_second_data = self._load_vel_data_by_imu_timestamps(vel_file, imu_timestamps, 3)
        common_seconds = set(imu_second_data.keys()) & set(vel_second_data.keys())
        print(f"Found {len(common_seconds)} seconds with complete 100Hz data")
        pos_dict = load_truePos(pos_file)
        phi_dict = load_matrix_file(matrix_files["phi"], 15, 15) if os.path.exists(matrix_files["phi"]) else {}
        H_dict = load_matrix_file(matrix_files["H"], 3, 15) if os.path.exists(matrix_files["H"]) else {}
        R_dict = load_matrix_file(matrix_files["R"], 3, 3) if os.path.exists(matrix_files["R"]) else {}
        P_dict = load_matrix_file(matrix_files["P"], 15, 15) if os.path.exists(matrix_files["P"]) else {}
        state_dict = load_matrix_file(matrix_files["state"], 15, 1) if os.path.exists(matrix_files["state"]) else {}
        inno_dict = load_matrix_file(matrix_files["inno"], 3, 1) if os.path.exists(matrix_files["inno"]) else {}
        Q_dict = load_matrix_file(matrix_files["Q"], 15, 15) if os.path.exists(matrix_files["Q"]) else {}
        print(f"Matrix data loaded: pos({len(pos_dict)}), phi({len(phi_dict)}), H({len(H_dict)}), R({len(R_dict)}), P({len(P_dict)}), state({len(state_dict)}), inno({len(inno_dict)}), Q({len(Q_dict)})")

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

        matrix_times = set(pos_dict.keys()) & set(phi_dict.keys()) & set(H_dict.keys()) & \
                       set(R_dict.keys()) & set(P_dict.keys()) & set(state_dict.keys()) & \
                       set(inno_dict.keys()) & set(Q_dict.keys())
        valid_seconds = common_seconds & matrix_times
        print(f"Found {len(valid_seconds)} seconds with aligned data")
        samples = []
        for second in sorted(valid_seconds):
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
                "phi": phi_dict[second],
                "H": H_dict[second],
                "R": R_dict[second],
                "P": P_dict[second],
                "stateX": stateX_mat,
                "Inno": inno_dict[second],
                "Q": Q_dict[second],
            }
            samples.append(sample)
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
                second = int(full_time)
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
                second = int(full_time)
                if vel_start_second is None:
                    vel_start_second = second
                    vel_end_second = second
                else:
                    vel_start_second = min(vel_start_second, second)
                    vel_end_second = max(vel_end_second, second)
        print(f"Total Vel data points loaded: {len(vel_data_all)}")
        vel_data_by_second = {}
        for vel_time, vel_data in vel_data_all.items():
            second = int(vel_time)
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
        return {
            "imu": imu_tensor,
            "time": torch.tensor(sample["time"], dtype=torch.float64),
            "matrix_time": torch.tensor(sample["matrix_time"], dtype=torch.float64),
            "pos_true": torch.tensor(sample["pos_true"], dtype=torch.float64),
            "phi": torch.tensor(sample["phi"], dtype=torch.float64),
            "H": torch.tensor(sample["H"], dtype=torch.float64),
            "R": torch.tensor(sample["R"], dtype=torch.float64),
            "P": torch.tensor(sample["P"], dtype=torch.float64),
            "stateX": torch.tensor(sample["stateX"], dtype=torch.float64),
            "Inno": torch.tensor(sample["Inno"], dtype=torch.float64),
            "Q": torch.tensor(sample["Q"], dtype=torch.float64),
        }


class IMUDataset(Dataset):
    def __init__(self, data_dir, seq_len=100):
        self.multi_dataset = MultiDatasetIMU([data_dir], seq_len)
    def __len__(self):
        return len(self.multi_dataset)
    def __getitem__(self, idx):
        return self.multi_dataset[idx]


def kalman_filter(phi, H, R, P, Inno, stateX, Q_diag, Q_origin):
    """Pure-Torch differentiable single-step Kalman update (batch-aware).
    Inputs (batch first):
      phi: (B, n, n)
      H:   (B, m, n)
      R:   (B, m, m)
      P:   (B, n, n)
      Inno:(B, m, 1)
      stateX:(B, n, 1)
      Q_diag:(B, n, n) -- process noise (already expanded to full matrix)
    Returns:
      pos_predict: (B, 3) -- first 3 components of updated state
    This implementation uses torch operations only and adds small jitter to S diagonal
    for numerical stability. Uses torch.linalg.solve when possible.
    """
    eps = 1e-8
    device = phi.device
    dtype = phi.dtype
    # 强制使用 double 精度进行卡尔曼内部计算，避免 float/double mismatch（保留 autograd）
    try:
        target_dtype = torch.float64
        phi = phi.to(device=device, dtype=target_dtype)
        H = H.to(device=device, dtype=target_dtype)
        R = R.to(device=device, dtype=target_dtype)
        P = P.to(device=device, dtype=target_dtype)
        Inno = Inno.to(device=device, dtype=target_dtype)
        if stateX is not None:
            stateX = stateX.to(device=device, dtype=target_dtype)
        Q_diag = Q_diag.to(device=device, dtype=target_dtype)
        Q_origin = Q_origin.to(device=device, dtype=target_dtype)
        dtype = target_dtype
    except Exception:
        pass

    # Predict covariance: P_prior = phi * P * phi^T + Q_diag
    try:
            # 支持 Q_diag 为 (B, n)（表示对角缩放因子）或 (B, n, n)（表示与 Q_origin 做按元素缩放）
        if Q_diag.dim() == 2:
            # Q_diag 为 (B, n) 时，视为对 Q_origin 对角线的缩放因子。
            # 为保留 autograd 路径，避免对张量做 in-place 写操作。
            # 提取 Q_origin 的对角并按 Q_diag 缩放，然后重建一个对角矩阵并与 Q_origin 的非对角部分合并。
            # diag_orig: (B, n)
            try:
                diag_orig = Q_origin.diagonal(offset=0, dim1=-2, dim2=-1)
            except Exception:
                # 兼容较旧 PyTorch 版本的接口
                diag_orig = torch.stack([torch.diagonal(m) for m in Q_origin], dim=0)
            # 构造缩放后的对角线（保留 autograd 路径）
            scaled_diag = diag_orig * Q_diag
            # 构造对角矩阵 (B, n, n) - 现在整个 Q 矩阵都与 Q_diag 相关
            Q_effective = torch.diag_embed(scaled_diag)
            # 注意：这里我们不再保留原始的非对角部分，让网络学习完整的 Q 矩阵
            # --- 调试：检查计算图 ---
            if torch.is_grad_enabled() and Q_diag.requires_grad:
                # 确保计算图连接
                pass
        elif Q_diag.dim() == 3 and Q_diag.shape == Q_origin.shape:
            # 如果提供的是与 Q_origin 形状相同的矩阵，则按元素相乘（保留用户意图）
            Q_effective = Q_origin * Q_diag
        else:
            # 明确报错，避免静默兼容导致难以发现的问题
            raise ValueError(f"Unexpected Q_diag shape: {tuple(Q_diag.shape)}; expected (B,n) or (B,n,n) matching Q_origin {tuple(Q_origin.shape)}")
    except Exception as e:
        # 抛出运行时错误以便上层可以捕获并定位问题
        raise RuntimeError(f"Failed to construct Q_effective from Q_diag: {e}")

    # 支持 H/R 为非批量（广播）
    if H.dim() == 2:
        H = H.unsqueeze(0).expand(P.shape[0], -1, -1)
    if R.dim() == 2:
        R = R.unsqueeze(0).expand(P.shape[0], -1, -1)

    # 支持 phi 为非批量（广播），并计算先验协方差 P_prior = phi P phi^T + Q_effective
    if phi.dim() == 2:
        phi = phi.unsqueeze(0).expand(P.shape[0], -1, -1)
        # 先验协方差（恢复原有公式）
    P_pred = torch.matmul(torch.matmul(phi, P), phi.transpose(-2, -1)) + Q_effective
        # 预测 state（若需要）：state_pred = phi @ stateX
    try:
        if stateX is not None and stateX.dim() == 3:
            state_pred = torch.matmul(phi, stateX)
    except Exception:
        pass

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

    # 更新 state
    state_update = torch.matmul(K, Inno)  # (B, n, 1)
    state_post = state_pred + state_update

    # ====== 合并后的精简 DEBUG 打印 (仅打印第一个样本的前3维) ======
    if stateX is not None and 'state_pred' in locals():
        try:
            with torch.no_grad():
                # 提取数据
                s_ori = stateX[0, :3, 0].cpu().numpy()
                s_pre = state_pred[0, :3, 0].cpu().numpy()
                s_upd = state_update[0, :3, 0].cpu().numpy()
                s_pst = state_post[0, :3, 0].cpu().numpy()
                
                # 格式化函数
                fmt = lambda x: "[" + ",".join([f"{v:.4f}" for v in x]) + "]"
                
                # 单行打印核心链路
                print(f"DEBUG KF [Pos]: Ori:{fmt(s_ori)} -> Pre:{fmt(s_pre)} + Upd:{fmt(s_upd)} = Pst:{fmt(s_pst)}")
        except Exception as e:
            print(f"DEBUG KF: Print failed ({e})")
    # ==========================================================

    # Joseph form 保证数值稳定并保持半正定
    I = torch.eye(P.shape[-1], device=device, dtype=dtype).unsqueeze(0).expand(P.shape[0], -1, -1)
    KH = torch.matmul(K, H)  # (B, n, n)
    temp = I - KH
    P_post = torch.matmul(torch.matmul(temp, P_pred), temp.transpose(-2, -1)) + torch.matmul(torch.matmul(K, R), K.transpose(-2, -1))

    # 若需要将 state_post 变回 (B, n) 形式
    state_post = state_post.squeeze(-1)

    # 仅返回位置向量的前三个分量作为 pos_predict (B,3)
    pos_predict = state_post.squeeze(-1)[:, 0:3]
    return pos_predict
