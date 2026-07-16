"""
数据读取测试代码 - 完全还原原始数据读取逻辑
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import os


def load_truePos(file_path):
    """读取位置真值文件"""
    data_dict = {}
    try:
        with open(file_path, "r") as f:
            line_count = 0
            header_found = False
            
            for line in f:
                line_count += 1
                line = line.strip()
                
                # 跳过空行和注释行
                if not line or line.startswith('#') or 'Profile:' in line or 'Source:' in line:
                    continue
                
                # 寻找数据表头行
                if 'Week' in line and 'GPSTime' in line and not header_found:
                    print(f"  找到表头行 {line_count}: Week GPSTime ...")
                    header_found = True
                    continue
                
                # 寻找单位行
                if 'week' in line and '(sec)' in line:
                    print(f"  找到单位行 {line_count}")
                    continue
                
                # 跳过未找到表头前的行
                if not header_found:
                    continue
                
                # 解析数据行
                values = line.split()
                if len(values) < 15:  # 至少需要有基本的位置信息
                    if line_count <= 20:  # 只显示前20行的警告
                        print(f"  第{line_count}行数据不足: 需要至少15个值，实际{len(values)}个")
                    continue
                
                try:
                    # GPS时间解析 (Week + GPSTime)
                    week = int(values[0])
                    gps_time = float(values[1])
                    
                    # 转换为连续时间戳 (简化处理，直接使用GPSTime)
                    time_stamp = gps_time
                    
                    # 提取位置信息 (X-LL, Y-LL, Z-LL 在第10、11、12列)
                    if len(values) >= 13:
                        x_ll = float(values[9])  # X-LL
                        y_ll = float(values[10])  # Y-LL  
                        z_ll = float(values[11])  # Z-LL
                        
                        pos = np.array([x_ll, y_ll, z_ll])
                        data_dict[time_stamp] = pos
                        
                except (ValueError, IndexError) as e:
                    if line_count <= 20:
                        print(f"  第{line_count}行解析错误: {e}")
                    continue
            
            print(f"  成功读取 {len(data_dict)} 条位置数据 (总共{line_count}行)")
                  # 显示时间范围
        if data_dict:
            times = list(data_dict.keys())
            print(f"  时间范围: {min(times)} 到 {max(times)}")
        
    except Exception as e:
        print(f"  读取位置文件出错: {e}")
        
    return data_dict


def load_matrix_file(file_path, rows, cols):
    """读取矩阵文件 - 处理多行矩阵格式"""
    data_dict = {}
    try:
        with open(file_path, "r") as f:
            header = f.readline().strip()
            # print(f"  矩阵文件表头: {header}")  # 简化输出
            
            lines = f.readlines()
            i = 0
            matrix_count = 0
            
            while i < len(lines):
                line = lines[i].strip()
                
                # 跳过空行
                if not line:
                    i += 1
                    continue
                
                # 尝试解析时间戳行
                values = line.split()
                
                # 检查是否是时间戳行（第一个值是纯数字，且后面跟着矩阵数据）
                try:
                    time_stamp = int(values[0])
                    
                    # 检查这行是否包含时间戳+矩阵第一行的数据
                    if len(values) > 1:
                        # 格式1: 时间戳和第一行矩阵数据在同一行
                        matrix_data = []
                        
                        # 第一行数据（从当前行提取，跳过时间戳）
                        first_row = [float(v) for v in values[1:]]
                        if len(first_row) >= cols:
                            matrix_data.extend(first_row[:cols])
                        else:
                            # 数据不足，用0填充
                            matrix_data.extend(first_row + [0.0] * (cols - len(first_row)))
                        
                        # 读取后续行
                        for row_idx in range(1, rows):
                            i += 1
                            if i < len(lines):
                                row_line = lines[i].strip()
                                if row_line:
                                    row_values = [float(v) for v in row_line.split()]
                                    if len(row_values) >= cols:
                                        matrix_data.extend(row_values[:cols])
                                    else:
                                        # 数据不足，用0填充
                                        matrix_data.extend(row_values + [0.0] * (cols - len(row_values)))
                                else:
                                    # 空行，用0填充
                                    matrix_data.extend([0.0] * cols)
                            else:
                                # 文件结束，用0填充
                                matrix_data.extend([0.0] * cols)
                        
                        # 创建矩阵
                        if len(matrix_data) >= rows * cols:
                            matrix = np.array(matrix_data[:rows * cols]).reshape(rows, cols)
                            data_dict[time_stamp] = matrix
                            matrix_count += 1
                    
                    else:
                        # 格式2: 时间戳单独一行，后面是矩阵数据
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
                            matrix_count += 1
                
                except ValueError:
                    # 不是有效的时间戳行，跳过
                    pass
                
                i += 1
            
            print(f"  成功读取 {len(data_dict)} 条矩阵数据")
            
            # if len(data_dict) > 0:  # 简化输出
            #     times = list(data_dict.keys())
            #     print(f"  时间范围: {min(times)} 到 {max(times)}")
            
    except Exception as e:
        print(f"  读取文件出错: {e}")
        import traceback
        traceback.print_exc()
        
    return data_dict


class MultiDatasetIMU(Dataset):
    def __init__(self, data_dirs, seq_len=100):
        """
        多数据集训练类
        data_dirs: list of data directories
        seq_len: 滑动窗口长度 (100Hz数据的窗口长度)
        """
        self.seq_len = seq_len
        self.all_samples = []
        
        for data_dir in data_dirs:
            print(f"Loading dataset from: {data_dir}")
            samples = self._load_single_dataset(data_dir)
            self.all_samples.extend(samples)
            print(f"Added {len(samples)} samples from {data_dir}")
        
        print(f"Total samples across all datasets: {len(self.all_samples)}")
    
    def _load_single_dataset(self, data_dir):
        """加载单个数据集"""
        # 文件路径
        imu_file = os.path.join(data_dir, "imu.txt")
        vel_file = os.path.join(data_dir, "Vel.txt")
        
        # 查找位置真值文件
        pos_files = [f for f in os.listdir(data_dir) 
                    if f.lower().endswith((".imr.ref", ".pos", ".ref"))]
        if not pos_files:
            raise FileNotFoundError(f"No position reference file found in {data_dir}")
        pos_file = os.path.join(data_dir, pos_files[0])
        
        # 矩阵文件路径
        matrix_files = {
            "phi": os.path.join(data_dir, "Phi.txt"),
            "H": os.path.join(data_dir, "H.txt"),
            "R": os.path.join(data_dir, "R.txt"),
            "P": os.path.join(data_dir, "StateP.txt"),
            "state": os.path.join(data_dir, "StateX.txt"),
            "inno": os.path.join(data_dir, "Inno.txt"),
            "Q": os.path.join(data_dir, "Q.txt")
        }
        
        # 读取高频数据并按整数秒分组，以IMU数据时间为准
        imu_second_data, imu_timestamps = self._load_and_group_imu_data(imu_file, 6)  # 6维IMU
        vel_second_data = self._load_vel_data_by_imu_timestamps(vel_file, imu_timestamps, 3)  # 3维速度
        
        # 找到IMU和Vel都有完整100个数据的共同整数秒
        common_seconds = set(imu_second_data.keys()) & set(vel_second_data.keys())
        print(f"Found {len(common_seconds)} seconds with complete 100Hz data")
        
        # 读取矩阵数据 (1Hz)
        pos_dict = load_truePos(pos_file)
        phi_dict = load_matrix_file(matrix_files["phi"], 15, 15) if os.path.exists(matrix_files["phi"]) else {}
        H_dict = load_matrix_file(matrix_files["H"], 3, 15) if os.path.exists(matrix_files["H"]) else {}
        R_dict = load_matrix_file(matrix_files["R"], 3, 3) if os.path.exists(matrix_files["R"]) else {}
        P_dict = load_matrix_file(matrix_files["P"], 15, 15) if os.path.exists(matrix_files["P"]) else {}
        state_dict = load_matrix_file(matrix_files["state"], 15, 1) if os.path.exists(matrix_files["state"]) else {}
        inno_dict = load_matrix_file(matrix_files["inno"], 3, 1) if os.path.exists(matrix_files["inno"]) else {}
        Q_dict = load_matrix_file(matrix_files["Q"], 15, 15) if os.path.exists(matrix_files["Q"]) else {}
        
        print(f"Matrix data loaded: pos({len(pos_dict)}), phi({len(phi_dict)}), H({len(H_dict)}), R({len(R_dict)}), P({len(P_dict)}), state({len(state_dict)}), inno({len(inno_dict)}), Q({len(Q_dict)})")
        
        # 时间对齐：找到既有完整高频数据又有矩阵数据的整数秒
        matrix_times = set(pos_dict.keys()) & set(phi_dict.keys()) & set(H_dict.keys()) & \
                      set(R_dict.keys()) & set(P_dict.keys()) & set(state_dict.keys()) & \
                      set(inno_dict.keys()) & set(Q_dict.keys())
        
        # 最终有效的时间点：既有100Hz数据又有矩阵数据
        valid_seconds = common_seconds & matrix_times
        print(f"Found {len(valid_seconds)} seconds with aligned data")
        
        # 生成样本
        samples = []
        for second in sorted(valid_seconds):
            # 合并IMU和Vel数据：每秒100个9维数据点
            imu_data_100 = imu_second_data[second]  # (100, 6)
            vel_data_100 = vel_second_data[second]  # (100, 3)
            
            # 合并为9维数据
            combined_data = []
            for i in range(self.seq_len):  # seq_len = 100
                combined = imu_data_100[i] + vel_data_100[i]  # 6+3=9维
                combined_data.append(combined)
            
            # 创建样本
            sample = {
                "imu": np.array(combined_data, dtype=np.float64),  # 9维传感器数据 (100, 9)
                "time": float(second),  # 当前整数秒
                "matrix_time": second, # 对应矩阵数据的时间
                "pos_true": pos_dict[second], # 真值位置
                "phi": phi_dict[second], # 状态转移矩阵
                "H": H_dict[second], # 观测矩阵
                "R": R_dict[second], # 观测噪声矩阵
                "P": P_dict[second], # 协方差矩阵
                "stateX": state_dict[second], # 上一时刻状态
                "Inno": inno_dict[second], # 观测新息
                "Q": Q_dict[second], # 原始模型计算的过程噪声
            }
            samples.append(sample)
            
        return samples
    
    def _load_and_group_imu_data(self, imu_file, data_dim):
        """
        读取IMU数据并按整数秒分组，以IMU时间为准
        返回: (dict[int_second] = [[data1], [data2], ..., [data100]], 
                dict[int_second] = [timestamp1, timestamp2, ..., timestamp100])
        只保留恰好有100个数据点的整数秒
        """
        second_groups = {}
        second_timestamps = {}
        
        with open(imu_file, "r") as f:
            header = f.readline()  # 跳过表头
            
            for line in f:
                values = line.strip().split()
                if len(values) < data_dim + 1:
                    continue
                
                # 提取完整时间戳和整数秒部分
                full_time = float(values[0])
                second = int(full_time)
                data_values = [float(v) for v in values[1:data_dim+1]]
                
                # 按整数秒分组存储数据和时间戳
                if second not in second_groups:
                    second_groups[second] = []
                    second_timestamps[second] = []
                
                second_groups[second].append(data_values)
                second_timestamps[second].append(full_time)
        
        # 只保留恰好有100个数据点的整数秒
        filtered_groups = {}
        filtered_timestamps = {}
        
        for second, data_list in second_groups.items():
            if len(data_list) == self.seq_len:  # seq_len = 100
                filtered_groups[second] = data_list
                filtered_timestamps[second] = second_timestamps[second]
            else:
                pass  # 静默跳过不完整的秒
        
        return filtered_groups, filtered_timestamps
    
    def _load_vel_data_by_imu_timestamps(self, vel_file, imu_timestamps, data_dim):
        """
        根据IMU数据的时间戳筛选对应的Vel数据
        imu_timestamps: dict[int_second] = [timestamp1, timestamp2, ..., timestamp100]
        返回: dict[int_second] = [[data1], [data2], ..., [data100]]
        """
        # 首先读取所有Vel数据并找到起始时间
        vel_data_all = {}
        vel_start_second = None
        vel_end_second = None
        
        with open(vel_file, "r") as f:
            header = f.readline()  # 跳过表头
            
            for line in f:
                values = line.strip().split()
                if len(values) < data_dim + 1:
                    continue
                
                full_time = float(values[0])
                data_values = [float(v) for v in values[1:data_dim+1]]
                vel_data_all[full_time] = data_values
                
                # 记录Vel数据的时间范围
                second = int(full_time)
                if vel_start_second is None:
                    vel_start_second = second
                    vel_end_second = second
                else:
                    vel_start_second = min(vel_start_second, second)
                    vel_end_second = max(vel_end_second, second)
        
        print(f"Total Vel data points loaded: {len(vel_data_all)}")
        
        # 预先按秒分组Vel数据以加速查找
        vel_data_by_second = {}
        for vel_time, vel_data in vel_data_all.items():
            second = int(vel_time)
            if second not in vel_data_by_second:
                vel_data_by_second[second] = {}
            vel_data_by_second[second][vel_time] = vel_data
        
        # 根据IMU时间戳筛选Vel数据，只处理Vel数据时间范围内的IMU数据
        filtered_vel_groups = {}
        
        for second, imu_times in imu_timestamps.items():
            # 跳过Vel数据时间范围外的IMU数据
            if second < vel_start_second or second > vel_end_second:
                continue
                
            # 如果这一秒没有Vel数据，直接跳过
            if second not in vel_data_by_second:
                continue
            
            vel_data_for_second = []
            matched_count = 0
            
            # 获取这一秒的所有Vel数据
            second_vel_data = vel_data_by_second[second]
            
            for imu_time in imu_times:
                # 在这一秒的Vel数据中寻找最接近的时间戳
                best_match = None
                min_diff = float('inf')
                
                for vel_time in second_vel_data.keys():
                    time_diff = abs(vel_time - imu_time)
                    if time_diff < min_diff:
                        min_diff = time_diff
                        best_match = vel_time
                
                # 只有时间差小于容差才认为匹配成功
                if best_match is not None and min_diff < 0.002:  # 2ms容差
                    vel_data_for_second.append(second_vel_data[best_match])
                    matched_count += 1
                else:
                    # 如果找不到匹配的时间戳，用最近的数据或零填充
                    if vel_data_for_second:  # 如果已有数据，复制最后一个
                        vel_data_for_second.append(vel_data_for_second[-1])
                    else:  # 否则用零填充
                        vel_data_for_second.append([0.0] * data_dim)
            
            # 只有当Vel数据也恰好有100个时才保留
            if len(vel_data_for_second) == self.seq_len:
                filtered_vel_groups[second] = vel_data_for_second
        
        return filtered_vel_groups
    
    def __len__(self):
        return len(self.all_samples)
    
    def __getitem__(self, idx):
        sample = self.all_samples[idx]
        
        # 转换为PyTorch张量
        imu_tensor = torch.tensor(sample["imu"], dtype=torch.float64)
        # 网络输入要求 (in_chans, seq_len)
        imu_tensor = imu_tensor.permute(1, 0)  # (9, seq_len)
        
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


def test_data_loading():
    """测试数据读取功能"""
    # 数据目录列表 - 使用绝对路径
    base_path = r"F:\AAA-lulul\科研工作\毕业设计\A-KIT-main\A-KIT-main\data_lj"
    data_dirs = [
        os.path.join(base_path, "data013325"),
        os.path.join(base_path, "data065517"),
        os.path.join(base_path, "data090854"),
        os.path.join(base_path, "data212832")
    ]
    
    print(f"=== 检查数据路径 ===")
    for data_dir in data_dirs:
        if os.path.exists(data_dir):
            print(f"✓ {os.path.basename(data_dir)}: OK")
        else:
            print(f"✗ {os.path.basename(data_dir)}: 路径不存在")
    
    # 创建数据集
    try:
        dataset = MultiDatasetIMU(data_dirs, seq_len=100)
        
        print(f"\n=== 数据集基本信息 ===")
        print(f"总样本数量: {len(dataset)}")
        
        if len(dataset) == 0:
            print("没有找到有效数据，请检查数据路径和文件格式")
            return
        
        # 输出第一个样本的基本信息
        print(f"\n=== 第一个样本信息 ===")
        sample = dataset[0]
        print(f"时间戳: {sample['time'].item()}")
        print(f"IMU数据形状: {sample['imu'].shape}")
        print(f"位置真值: {sample['pos_true'].numpy()}")
        
        # 数据完整性检查
        print(f"\n=== 数据完整性检查 ===")
        shapes = {
            'imu': [],
            'pos_true': [],
            'phi': [],
            'H': [],
            'R': [],
            'P': [],
            'stateX': [],
            'Inno': [],
            'Q': []
        }
        
        for i in range(min(5, len(dataset))):
            sample = dataset[i]
            for key in shapes.keys():
                shapes[key].append(tuple(sample[key].shape))
        
        # 检查形状一致性
        for key, shape_list in shapes.items():
            unique_shapes = list(set(shape_list))
            if len(unique_shapes) == 1:
                print(f"  {key}: 形状一致 {unique_shapes[0]}")
            else:
                print(f"  {key}: 形状不一致! {unique_shapes}")
        
        print(f"\n=== 多数据集测试完成 ===")
        print(f"✅ 成功加载 {len(dataset)} 个样本")
            
    except Exception as e:
        print(f"数据加载过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_data_loading()
