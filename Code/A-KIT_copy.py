# 在导入任何可能初始化 OpenMP 的库之前设置环境变量，避免 libiomp5md.dll 重复初始化错误
import os
# 不推荐的临时解决方法：允许重复加载 OpenMP 运行时以避免程序中断（可能有风险）
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")
# 限制线程数以降低不同库间线程冲突的概率
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
from datetime import datetime
import time


def main():
    # 1. 训练超参数
    batch_size = 32
    seq_len = 100  # IMU(t-99 ~ t) → 预测 Q(t)
    epochs = 50
    learning_rate = 1e-4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
     
    # 2. 多数据集路径 - 使用绝对路径
    base_path = r"F:\AAA-lulul\科研工作\毕业设计\A-KIT-main\A-KIT-main\data_lj"
    data_dirs = [
        os.path.join(base_path, "data013325"),
        os.path.join(base_path, "data065517"),
        os.path.join(base_path, "data090854"),
        os.path.join(base_path, "data212832")
    ]
    
    # 检查数据路径
    print(f"=== 检查数据路径 ===")
    for data_dir in data_dirs:
        if os.path.exists(data_dir):
            print(f"✓ {os.path.basename(data_dir)}: OK")
        else:
            print(f"✗ {os.path.basename(data_dir)}: 路径不存在")
    
    # 创建多数据集加载器
    dataset = MultiDatasetIMU(data_dirs, seq_len=seq_len)
    train_loader = DataLoader(dataset, batch_size=batch_size,
                             shuffle=True, drop_last=True)

    # 3. 构建模型
    model = Network(
        patch_size=10,
        filter_size=10,
        in_chans=9,
        embedding_size=64,
        num_heads=4,
        feedforward_expansion=128,
        k=1,
        blocklayers=2,
        output_size=15
    ).to(device).double()

    # 4. 优化器
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-6)
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"training_results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"训练结果将保存到: {output_dir}")
    
    # 训练监控变量
    train_losses = []
    epoch_times = []
    batch_losses = []
    learning_rates = []
    q_predictions = []
    position_errors = []
     
    # 5. 训练循环
    print(f"\n=== 开始训练 ===")
    print(f"设备: {device}")
    print(f"总样本数: {len(dataset)}")
    print(f"批次数: {len(train_loader)}")
    
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        model.train()
        total_loss = 0.0
        epoch_batch_losses = []
        epoch_pos_errors = []
        epoch_q_preds = []

        for batch_idx, batch in enumerate(train_loader):
            # 读取输入
            imu = batch["imu"].to(device).double()
            pos_true = batch["pos_true"].to(device).double()
             
            # 构造 Kalman 需要的矩阵
            phi = batch["phi"].to(device).double()
            H = batch["H"].to(device).double()
            R = batch["R"].to(device).double()
            Inno = batch["Inno"].to(device).double()
            P = batch["P"].to(device).double()
            stateX = batch["stateX"].to(device).double()
            Q_origin = batch["Q"].to(device).double()
             
            # 网络预测 Q 的对角线元素
            Q_diag = model(imu, P.unsqueeze(-1))
            Q_diag = torch.exp(Q_diag)  # 保证为正
            Q_matrix = torch.diag_embed(Q_diag)
             
            # 调用可微 Kalman Filter
            pos_predict = kalman_filter(phi, H, R, P, Inno, stateX, Q_matrix, Q_origin)
             
            # 计算位置 MSE
            loss = torch.mean((pos_predict - pos_true) ** 2)
            
            # 计算位置误差（米）
            pos_error = torch.sqrt(torch.mean((pos_predict - pos_true) ** 2, dim=1))
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 记录数据
            batch_loss = loss.item()
            total_loss += batch_loss
            epoch_batch_losses.append(batch_loss)
            epoch_pos_errors.extend(pos_error.cpu().detach().numpy())
            epoch_q_preds.append(Q_diag[0].cpu().detach().numpy())  # 记录第一个样本的Q预测
            
            # 每10个batch打印一次进度
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {batch_loss:.6f}")

        # 记录epoch统计
        avg_loss = total_loss / len(train_loader)
        epoch_time = time.time() - epoch_start_time
        
        train_losses.append(avg_loss)
        epoch_times.append(epoch_time)
        batch_losses.extend(epoch_batch_losses)
        learning_rates.append(optimizer.param_groups[0]['lr'])
        q_predictions.append(np.mean(epoch_q_preds, axis=0))
        position_errors.extend(epoch_pos_errors)

        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.6f}, Time: {epoch_time:.2f}s, Avg Pos Error: {np.mean(epoch_pos_errors):.3f}m")
        
        # 每10个epoch生成中间图表
        if (epoch + 1) % 10 == 0:
            generate_training_plots(train_losses, epoch_times, batch_losses, 
                                   learning_rates, q_predictions, position_errors, 
                                   output_dir, epoch + 1)

    total_training_time = time.time() - start_time
    print(f"\nTraining Finished. Total time: {total_training_time:.2f}s")
    
    # 生成最终的训练报告和图表
    generate_final_report(model, train_losses, epoch_times, batch_losses, 
                         learning_rates, q_predictions, position_errors, 
                         output_dir, total_training_time, dataset)


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.heads_dim = embedding_size // num_heads

        self.Query = nn.Linear(embedding_size, embedding_size, bias=False, dtype=torch.float64)
        self.Key = nn.Linear(embedding_size, embedding_size, bias=False, dtype=torch.float64)
        self.Value = nn.Linear(embedding_size, embedding_size, bias=False, dtype=torch.float64)
        self.fc_out = nn.Linear(embedding_size, embedding_size, bias=False, dtype=torch.float64)

    def forward(self, query, key, value):
        batch_size = query.shape[0]
        q, k, v = query.shape[1], key.shape[1], value.shape[1]

        V = self.Value(value)
        Q = self.Query(query)
        K = self.Key(key)

        # 分割为多头
        V = V.reshape(batch_size, v, self.num_heads, self.heads_dim)
        V = torch.einsum("bnhd->bhnd", V)
        Q = Q.reshape(batch_size, q, self.num_heads, self.heads_dim)
        Q = torch.einsum("bnhd->bhnd", Q)
        K = K.reshape(batch_size, k, self.num_heads, self.heads_dim)
        K = torch.einsum("bnhd->bhnd", K)

        # 计算注意力
        temp = torch.einsum("bhqd,bhkd->bhqk", [Q, K])
        Attention = torch.softmax(temp / torch.sqrt(torch.tensor(self.embedding_size, dtype=torch.float64)), dim=3)
        temp = torch.einsum("bhqk,bhkv->bhqv", [Attention, V])

        temp = temp.reshape(batch_size, q, self.num_heads * self.heads_dim)
        out = self.fc_out(temp)
        return out


class MAB(nn.Module):
    def __init__(self, embedding_size, num_heads, feedforward_expansion):
        super(MAB, self).__init__()
        self.MHA = MultiHeadAttention(embedding_size, num_heads)
        self.lnorm1 = nn.LayerNorm(embedding_size, dtype=torch.float64)
        self.lnorm2 = nn.LayerNorm(embedding_size, dtype=torch.float64)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_size, feedforward_expansion, dtype=torch.float64),
            nn.LeakyReLU(0.1),
            nn.Linear(feedforward_expansion, embedding_size, dtype=torch.float64),
        )

    def forward(self, X, Y):
        mha = self.MHA(X, Y, Y)
        x = self.lnorm1(mha + X)
        FeedForward = self.feed_forward(x)
        out = self.lnorm2(FeedForward + x)
        return out


class SAB(nn.Module):
    def __init__(self, embedding_size, num_heads, feedforward_expansion):
        super(SAB, self).__init__()
        self.mab = MAB(embedding_size, num_heads, feedforward_expansion)

    def forward(self, X):
        out = self.mab(X, X)
        return out


class PMA(nn.Module):
    def __init__(self, embedding_size, num_heads, feedforward_expansion, k):
        super(PMA, self).__init__()
        self.mab = MAB(embedding_size, num_heads, feedforward_expansion)
        self.S = torch.nn.init.kaiming_uniform_(
            torch.nn.Parameter(torch.zeros(k, embedding_size, dtype=torch.float64), requires_grad=True))
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_size, feedforward_expansion, dtype=torch.float64),
            nn.LeakyReLU(0.1),
            nn.Linear(feedforward_expansion, embedding_size, dtype=torch.float64),
        )

    def forward(self, Z):
        batch_size = Z.shape[0]
        out = self.mab(self.S.repeat(batch_size, 1, 1), self.feed_forward(Z))
        return out


class Decoder(nn.Module):
    def __init__(self, embedding_size, num_heads, feedforward_expansion, k):
        super(Decoder, self).__init__()
        self.sab = SAB(embedding_size, num_heads, feedforward_expansion)
        self.pma = PMA(embedding_size, num_heads, feedforward_expansion, k)

    def forward(self, x):
        out = self.sab(self.pma(x))
        return out


class SetTransformer(nn.Module):
    def __init__(self, embedding_size, num_heads, feedforward_expansion, k, blocklayers, output_size):
        super(SetTransformer, self).__init__()
        self.encoder_layers = nn.ModuleList([
            SAB(embedding_size, num_heads, feedforward_expansion)
            for _ in range(blocklayers)
        ])
        self.decoder_layers = nn.ModuleList([
            Decoder(embedding_size, num_heads, feedforward_expansion, k)
            for _ in range(blocklayers)
        ])
        self.fc_out = nn.Linear(embedding_size, output_size, dtype=torch.float64)

    def forward(self, x):
        for layer in self.encoder_layers:
            x = layer(x)
        for layer in self.decoder_layers:
            x = layer(x)
        return x


class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, filter_size, in_chans, embed_dim):
        super(PatchEmbedding, self).__init__()
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=filter_size, 
                             stride=patch_size, dtype=torch.float64)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x


class Network(nn.Module):
    def __init__(self, patch_size, filter_size, in_chans, embedding_size, num_heads, 
                 feedforward_expansion, k, blocklayers, output_size):
        super(Network, self).__init__()
        self.PE = PatchEmbedding(patch_size, filter_size, in_chans, embedding_size)
        self.set_transformer = SetTransformer(embedding_size, num_heads, feedforward_expansion, 
                                            k, blocklayers, output_size)
        self.FC_output = nn.Sequential(
            nn.Linear(embedding_size, 32, dtype=torch.float64),
            nn.LeakyReLU(0.1),
            nn.Dropout(p=0.1),
            nn.Linear(32, output_size, dtype=torch.float64),
            nn.ReLU(),
        )

    def forward(self, x, Q):
        x = self.PE(x)
        out = self.set_transformer(x)
        out = out.flatten(1)
        out = self.FC_output(out)
        
        # 避免零值
        mask = (out == 0)
        out = torch.where(mask, torch.tensor(1.0, dtype=torch.float64, device=out.device), out)
        
        return out


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
                    header_found = True
                    continue
                
                # 寻找单位行
                if 'week' in line and '(sec)' in line:
                    continue
                
                # 跳过未找到表头前的行
                if not header_found:
                    continue
                
                # 解析数据行
                values = line.split()
                if len(values) < 15:  # 至少需要有基本的位置信息
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
            
            lines = f.readlines()
            i = 0
            
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
                
                except ValueError:
                    # 不是有效的时间戳行，跳过
                    pass
                
                i += 1
            
            print(f"  成功读取 {len(data_dict)} 条矩阵数据")
            
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
            "state": os.path.join(data_dir, "StateX_new.txt"),
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


class IMUDataset(Dataset):
    """单数据集类（保持兼容性）"""
    def __init__(self, data_dir, seq_len=100):
        self.multi_dataset = MultiDatasetIMU([data_dir], seq_len)
    
    def __len__(self):
        return len(self.multi_dataset)
    
    def __getitem__(self, idx):
        return self.multi_dataset[idx]


def kalman_filter(phi, H, R, P, Inno, stateX, Q_diag, Q_origin):
    """
    卡尔曼滤波
    
    参数:
    phi      : (batch, 15, 15)       状态转移矩阵
    H        : (batch, m, 15)        观测矩阵
    R        : (batch, m, m)         观测噪声
    P        : (batch, 15, 15)       上一时刻协方差
    Inno     : (batch, m, 1)         观测新息
    stateX   : (batch, 15, 1)        上一时刻状态
    Q_diag   : (batch, 15, 15)       比例因子矩阵（网络预测）
    Q_origin : (batch, 15, 15)       原始模型计算的过程噪声
    
    返回:
    pos_predict : (batch, 3)         预测位置
    """
    
    # 调试输出 - 输入矩阵信息
    if torch.rand(1).item() < 0.05:  # 5%的概率输出调试信息
        print(f"\n{'='*60}")
        print(f"=== Kalman Filter 矩阵调试信息 ===")
        print(f"{'='*60}")
        
        # 输入矩阵维度信息
        print(f"输入矩阵维度:")
        print(f"  phi形状: {phi.shape}")
        print(f"  H形状: {H.shape}")
        print(f"  R形状: {R.shape}")
        print(f"  P形状: {P.shape}")
        print(f"  Inno形状: {Inno.shape}")
        print(f"  stateX形状: {stateX.shape}")
        print(f"  Q_diag形状: {Q_diag.shape}")
        print(f"  Q_origin形状: {Q_origin.shape}")
        
        # 第一个样本的矩阵内容（前几行/列）
        print(f"\n第一个样本的矩阵内容预览:")
        
        print(f"\nphi矩阵 (前5×5):")
        phi_preview = phi[0, :5, :5].detach().cpu().numpy()
        for i in range(5):
            row_str = "  " + " ".join([f"{val:8.4f}" for val in phi_preview[i]])
            print(row_str)
        
        print(f"\nH矩阵 (全部):")
        H_preview = H[0].detach().cpu().numpy()
        for i in range(H_preview.shape[0]):
            row_str = "  " + " ".join([f"{val:8.4f}" for val in H_preview[i, :8]])  # 只显示前8列
            if H_preview.shape[1] > 8:
                row_str += " ..."
            print(row_str)
        
        print(f"\nR矩阵 (全部):")
        R_preview = R[0].detach().cpu().numpy()
        for i in range(R_preview.shape[0]):
            row_str = "  " + " ".join([f"{val:8.4f}" for val in R_preview[i]])
            print(row_str)
        
        print(f"\nP矩阵 (前5×5):")
        P_preview = P[0, :5, :5].detach().cpu().numpy()
        for i in range(5):
            row_str = "  " + " ".join([f"{val:8.4f}" for val in P_preview[i]])
            print(row_str)
        
        print(f"\nInno向量:")
        Inno_preview = Inno[0, :, 0].detach().cpu().numpy()
        row_str = "  " + " ".join([f"{val:8.4f}" for val in Inno_preview])
        print(row_str)
        
        print(f"\nstateX向量 (前10个元素):")
        stateX_preview = stateX[0, :10, 0].detach().cpu().numpy()
        row_str = "  " + " ".join([f"{val:8.4f}" for val in stateX_preview])
        print(row_str)
        
        print(f"\nQ_diag矩阵对角线元素:")
        Q_diag_preview = torch.diag(Q_diag[0]).detach().cpu().numpy()
        row_str = "  " + " ".join([f"{val:8.6f}" for val in Q_diag_preview])
        print(row_str)
        
        print(f"\nQ_origin矩阵对角线元素:")
        Q_origin_preview = torch.diag(Q_origin[0]).detach().cpu().numpy()
        row_str = "  " + " ".join([f"{val:8.6f}" for val in Q_origin_preview])
        print(row_str)

    # 1. 状态预测
    x_pred = torch.matmul(phi, stateX)

    # 2. 协方差预测
    phi_T = phi.transpose(-1, -2)
    P_pred = torch.matmul(torch.matmul(phi, P), phi_T) + torch.matmul(Q_diag, Q_origin)

    # 3. 更新增益矩阵K、误差状态向量X
    H_T = H.transpose(-1, -2)
    # S = H P H^T + R
    S = torch.matmul(torch.matmul(H, P_pred), H_T) + R
    
    # 数值稳定处理（防止奇异）
    eps = 1e-6
    I_m = torch.eye(S.shape[-1], device=S.device, dtype=S.dtype).unsqueeze(0).expand_as(S)
    S = S + eps * I_m
    
    # S^-1
    S_inv = torch.linalg.inv(S)
    # K = P H^T S^-1
    K = torch.matmul(torch.matmul(P_pred, H_T), S_inv)

    # x_new = x_pred + K * Inno
    x_new = x_pred + torch.matmul(K, Inno)

    # 调试输出 - 中间计算结果和最终输出
    if torch.rand(1).item() < 0.05:  # 5%的概率输出调试信息
        print(f"\n中间计算结果:")
        
        print(f"\nx_pred (状态预测, 前10个元素):")
        x_pred_preview = x_pred[0, :10, 0].detach().cpu().numpy()
        row_str = "  " + " ".join([f"{val:8.4f}" for val in x_pred_preview])
        print(row_str)
        
        print(f"\nP_pred矩阵 (前5×5):")
        P_pred_preview = P_pred[0, :5, :5].detach().cpu().numpy()
        for i in range(5):
            row_str = "  " + " ".join([f"{val:8.4f}" for val in P_pred_preview[i]])
            print(row_str)
        
        print(f"\nS矩阵 (创新协方差):")
        S_preview = S[0].detach().cpu().numpy()
        for i in range(S_preview.shape[0]):
            row_str = "  " + " ".join([f"{val:8.4f}" for val in S_preview[i]])
            print(row_str)
        
        print(f"\nK矩阵 (卡尔曼增益, 前5行):")
        K_preview = K[0, :5, :].detach().cpu().numpy()
        for i in range(5):
            row_str = "  " + " ".join([f"{val:8.4f}" for val in K_preview[i]])
            print(row_str)
        
        print(f"\nx_new (更新后状态, 前10个元素):")
        x_new_preview = x_new[0, :10, 0].detach().cpu().numpy()
        row_str = "  " + " ".join([f"{val:8.4f}" for val in x_new_preview])
        print(row_str)
        
        print(f"\nx_new完整状态向量 (所有15个元素):")
        x_new_full = x_new[0, :, 0].detach().cpu().numpy()
        for i in range(15):
            print(f"  维度[{i:2d}]: {x_new_full[i]:12.6f}")
        
        print(f"{'='*60}")

    # 提取位置预测（假设位置在状态前3维）
    pos_predict = x_new[:, 0:3, 0]

    return pos_predict


def example_usage():
    """
    时间对齐逻辑说明：
    
    1. 高频数据 (IMU/Vel, 100Hz)：
       - 时间格式：215502.015646, 215502.025646, ..., 215502.995646
       - 每秒100个数据点，间隔0.01秒
    
    2. 矩阵数据 (1Hz)：
       - 时间格式：215502, 215503, 215504, ...
       - 每秒1个数据点
    
    3. 滑动窗口对齐：
       - 窗口长度：100 (对应1秒的高频数据)
       - 当前时刻：215505 (矩阵数据时间)
       - 窗口数据：215502.015646 到 215502.995646 的100个IMU/Vel数据点
       - 预测目标：215502时刻的Q矩阵
    """
    
    # 多数据集训练
    data_dirs = [
        "f:/Dataset/train/@@2020-05-20-013325/data013325",
        "f:/Dataset/train/@@2020-05-20-014026/data014026", 
        "f:/Dataset/train/@@2020-05-20-014727/data014727",
        "f:/Dataset/train/@@2020-05-20-015428/data015428",
        "f:/Dataset/train/@@2020-05-20-020129/data020129"
    ]
    
    dataset = MultiDatasetIMU(data_dirs, seq_len=100)
    print(f"Total aligned samples: {len(dataset)}")


def check_time_alignment(data_dir):
    """检查数据时间对齐情况"""
    required_files = ["imu.txt", "Vel.txt", "Inno.txt", "Phi.txt", "H.txt", 
                     "R.txt", "StateP.txt", "StateX_new.txt", "Q.txt"]
    
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


def generate_training_plots(train_losses, epoch_times, batch_losses, 
                          learning_rates, q_predictions, position_errors, 
                          output_dir, current_epoch):
    """生成训练过程的可视化图表"""
    
    # 创建子图
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Training Progress - Epoch {current_epoch}', fontsize=16)
    
    # 1. Loss收敛曲线
    axes[0, 0].plot(range(1, len(train_losses) + 1), train_losses, 'b-', linewidth=2)
    axes[0, 0].set_title('Training Loss Convergence')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_yscale('log')
    
    # 2. 每个epoch的训练时间
    axes[0, 1].bar(range(1, len(epoch_times) + 1), epoch_times, color='green', alpha=0.7)
    axes[0, 1].set_title('Training Time per Epoch')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Time (seconds)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 位置误差分布
    if position_errors:
        axes[0, 2].hist(position_errors, bins=50, color='red', alpha=0.7, edgecolor='black')
        axes[0, 2].set_title('Position Error Distribution')
        axes[0, 2].set_xlabel('Position Error (meters)')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].axvline(np.mean(position_errors), color='blue', linestyle='--', 
                          label=f'Mean: {np.mean(position_errors):.3f}m')
        axes[0, 2].legend()
    
    # 4. Q矩阵对角线元素演化（选择前几个元素）
    if q_predictions:
        q_array = np.array(q_predictions)
        for i in range(min(5, q_array.shape[1])):  # 显示前5个元素
            axes[1, 0].plot(range(1, len(q_predictions) + 1), q_array[:, i], 
                           label=f'Q_{i+1}', linewidth=2)
        axes[1, 0].set_title('Q Matrix Diagonal Elements Evolution')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Q Value')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_yscale('log')
    
    # 5. 批次loss变化（最近1000个批次）
    recent_batch_losses = batch_losses[-1000:] if len(batch_losses) > 1000 else batch_losses
    axes[1, 1].plot(recent_batch_losses, 'orange', alpha=0.7, linewidth=1)
    axes[1, 1].set_title('Recent Batch Losses')
    axes[1, 1].set_xlabel('Recent Batches')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. 学习率变化
    axes[1, 2].plot(range(1, len(learning_rates) + 1), learning_rates, 'purple', linewidth=2)
    axes[1, 2].set_title('Learning Rate Schedule')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Learning Rate')
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].set_yscale('log')
    
    plt.tight_layout()
    
    # 保存图表
    plot_path = os.path.join(output_dir, f'training_progress_epoch_{current_epoch}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"训练图表已保存到: {plot_path}")


def generate_final_report(model, train_losses, epoch_times, batch_losses, 
                         learning_rates, q_predictions, position_errors, 
                         output_dir, total_training_time, dataset):
    """生成最终训练报告"""
    
    # 1. 保存模型
    model_path = os.path.join(output_dir, 'final_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'train_losses': train_losses,
        'total_time': total_training_time,
        'final_loss': train_losses[-1] if train_losses else 0
    }, model_path)
    
    # 2. 生成最终图表
    generate_training_plots(train_losses, epoch_times, batch_losses, 
                          learning_rates, q_predictions, position_errors, 
                          output_dir, len(train_losses))
    
    # 3. 生成详细的收敛分析图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Detailed Training Analysis', fontsize=16)
    
    # Loss收敛详细分析
    axes[0, 0].plot(range(1, len(train_losses) + 1), train_losses, 'b-', linewidth=2, label='Training Loss')
    if len(train_losses) > 10:
        # 添加移动平均
        window_size = max(1, len(train_losses) // 10)
        moving_avg = np.convolve(train_losses, np.ones(window_size)/window_size, mode='valid')
        axes[0, 0].plot(range(window_size, len(train_losses) + 1), moving_avg, 'r--', 
                       linewidth=2, label=f'Moving Average ({window_size})')
    axes[0, 0].set_title('Training Loss Convergence Analysis')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_yscale('log')
    
    # 位置误差统计
    if position_errors:
        axes[0, 1].boxplot(position_errors, patch_artist=True)
        axes[0, 1].set_title('Position Error Statistics')
        axes[0, 1].set_ylabel('Position Error (meters)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 添加统计信息
        stats_text = f'Mean: {np.mean(position_errors):.4f}m\n'
        stats_text += f'Std: {np.std(position_errors):.4f}m\n'
        stats_text += f'Median: {np.median(position_errors):.4f}m\n'
        stats_text += f'95%ile: {np.percentile(position_errors, 95):.4f}m'
        axes[0, 1].text(0.02, 0.98, stats_text, transform=axes[0, 1].transAxes, 
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))
    
    # Q矩阵热力图（最后一个epoch）
    if q_predictions:
        q_final = q_predictions[-1]
        q_matrix_diag = np.diag(q_final)
        im = axes[1, 0].imshow(q_matrix_diag.reshape(-1, 1), cmap='viridis', aspect='auto')
        axes[1, 0].set_title('Final Q Matrix Diagonal Elements')
        axes[1, 0].set_ylabel('Element Index')
        plt.colorbar(im, ax=axes[1, 0])
    
    # 训练效率分析
    if epoch_times:
        axes[1, 1].plot(range(1, len(epoch_times) + 1), epoch_times, 'g-', linewidth=2)
        axes[1, 1].set_title('Training Efficiency')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Time per Epoch (seconds)')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 添加平均时间线
        avg_time = np.mean(epoch_times)
        axes[1, 1].axhline(y=avg_time, color='red', linestyle='--', 
                          label=f'Average: {avg_time:.2f}s')
        axes[1, 1].legend()
    
    plt.tight_layout()
    
    # 保存详细分析图
    analysis_path = os.path.join(output_dir, 'detailed_analysis.png')
    plt.savefig(analysis_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. 生成文本报告
    report_path = os.path.join(output_dir, 'training_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("A-KIT 训练报告\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("训练配置:\n")
        f.write(f"- 数据集样本数: {len(dataset)}\n")
        f.write(f"- 训练轮次: {len(train_losses)}\n")
        f.write(f"- 总训练时间: {total_training_time:.2f} 秒\n")
        f.write(f"- 平均每轮时间: {np.mean(epoch_times):.2f} 秒\n\n")
        
        f.write("训练结果:\n")
        f.write(f"- 初始Loss: {train_losses[0]:.6f}\n")
        f.write(f"- 最终Loss: {train_losses[-1]:.6f}\n")
        f.write(f"- Loss下降: {(train_losses[0] - train_losses[-1])/train_losses[0]*100:.2f}%\n\n")
        
        if position_errors:
            f.write("位置误差统计:\n")
            f.write(f"- 平均误差: {np.mean(position_errors):.4f} 米\n")
            f.write(f"- 标准差: {np.std(position_errors):.4f} 米\n")
            f.write(f"- 中位数误差: {np.median(position_errors):.4f} 米\n")
            f.write(f"- 95%分位数: {np.percentile(position_errors, 95):.4f} 米\n")
            f.write(f"- 99%分位数: {np.percentile(position_errors, 99):.4f} 米\n\n")
        
        if q_predictions:
            f.write("Q矩阵演化:\n")
            q_initial = q_predictions[0]
            q_final = q_predictions[-1]
            for i in range(min(5, len(q_final))):
                f.write(f"- Q[{i+1}]: {q_initial[i]:.6f} → {q_final[i]:.6f}\n")
    
    print(f"\n=== 训练完成报告 ===")
    print(f"模型已保存到: {model_path}")
    print(f"详细分析图: {analysis_path}")
    print(f"训练报告: {report_path}")
    print(f"总训练时间: {total_training_time:.2f}秒")
    print(f"最终Loss: {train_losses[-1]:.6f}")
    if position_errors:
        print(f"平均位置误差: {np.mean(position_errors):.4f}米")


if __name__ == "__main__":
    main()