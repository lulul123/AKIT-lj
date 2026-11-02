"""A-KIT 早期独立训练实现，仅保留用于历史对照。

该文件把数据读取、模型和训练逻辑放在一个脚本中，路径与参数也较多为硬编码，不属于
当前标准化训练入口。新实验应使用 A-KIT_lj_agent.py 和 configs 下的 JSON；除非进行
历史结果复现，不建议继续在本文件上扩展功能。
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import os
import math


def sample_window_key(timestamp):
    """Return k for the left-closed, right-open sample window [k-1, k)."""
    return math.floor(timestamp) + 1


def main():
    # 1. 训练超参数
    batch_size = 32
    seq_len = 100  # IMU(t-99 ~ t) → 预测 Q(t)
    epochs = 50
    learning_rate = 1e-4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
     
    # 2. 多数据集路径
    data_dirs = [
        "data_lj/data013325",
        "data_lj/data065517", 
        "data_lj/data090854",
        "data_lj/data122041",
        "data_lj/data212832"
    ]
    
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
     
    # 5. 训练循环
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
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
             
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}]  Loss: {total_loss:.6f}")

    print("Training Finished.")


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
    with open(file_path, "r") as f:
        for line in f:
            if not line.strip() or any(c.isalpha() for c in line):
                continue
            values = line.strip().split()
            if len(values) < 4:
                continue
            time = float(values[0])
            pos = np.array([float(values[1]), float(values[2]), float(values[3])])
            data_dict[time] = pos
    return data_dict


def load_matrix_file(file_path, rows, cols):
    """读取矩阵文件"""
    data_dict = {}
    with open(file_path, "r") as f:
        header = f.readline()
        for line in f:
            values = line.strip().split()
            if len(values) < 1 + rows * cols:
                continue
            time = int(values[0])
            matrix_values = [float(v) for v in values[1:]]
            matrix = np.array(matrix_values).reshape(rows, cols)
            data_dict[time] = matrix
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
        
        # 读取高频数据并按整数秒分组，确保每秒恰好100个数据
        imu_second_data = self._load_and_group_high_freq_data(imu_file, 6)  # 6维IMU
        vel_second_data = self._load_and_group_high_freq_data(vel_file, 3)  # 3维速度
        
        # 找到IMU和Vel都有完整100个数据的共同整数秒
        common_seconds = set(imu_second_data.keys()) & set(vel_second_data.keys())
        print(f"Found {len(common_seconds)} seconds with complete 100Hz data")
        
        # 读取矩阵数据 (1Hz)
        pos_dict = load_truePos(pos_file)
        phi_dict = load_matrix_file(matrix_files["phi"], 15, 15)
        H_dict = load_matrix_file(matrix_files["H"], 3, 15)
        R_dict = load_matrix_file(matrix_files["R"], 3, 3)
        P_dict = load_matrix_file(matrix_files["P"], 15, 15)
        state_dict = load_matrix_file(matrix_files["state"], 15, 1)
        inno_dict = load_matrix_file(matrix_files["inno"], 3, 1)
        Q_dict = load_matrix_file(matrix_files["Q"], 15, 15)
        
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
    
    def _load_and_group_high_freq_data(self, file_path, data_dim):
        """
        读取高频数据并按整数秒分组
        返回: dict[int_second] = [[data1], [data2], ..., [data100]]
        只保留恰好有100个数据点的整数秒
        """
        second_groups = {}
        
        with open(file_path, "r") as f:
            header = f.readline()  # 跳过表头
            
            for line in f:
                values = line.strip().split()
                if len(values) < data_dim + 1:
                    continue
                
                # 提取整数秒部分
                full_time = float(values[0])
                second = sample_window_key(full_time)
                data_values = [float(v) for v in values[1:data_dim+1]]
                
                # 按整数秒分组
                if second not in second_groups:
                    second_groups[second] = []
                second_groups[second].append(data_values)
        
        # 只保留恰好有100个数据点的整数秒
        filtered_groups = {}
        for second, data_list in second_groups.items():
            if len(data_list) == self.seq_len:  # seq_len = 100
                filtered_groups[second] = data_list
            else:
                print(f"Warning: Second {second} has {len(data_list)} data points, expected {self.seq_len}. Skipping.")
        
        return filtered_groups

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
    可微分的卡尔曼滤波器
    
    参数:
    phi      : (batch, 15, 15)       状态转移矩阵
    H        : (batch, m, 15)        观测矩阵
    R        : (batch, m, m)         观测噪声
    P        : (batch, 15, 15)       上一时刻协方差
    Inno     : (batch, m, 1)         观测新息
    stateX   : (batch, 15, 1)        上一时刻状态
    Q_diag   : (batch, 15, 15)       过程噪声矩阵（网络预测）
    Q_origin : (batch, 15, 15)       原始模型计算的过程噪声
    
    返回:
    pos_predict : (batch, 3)         预测位置
    """
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

    # 4. 提取位置预测（假设位置在状态前3维）
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
                     "R.txt", "StateP.txt", "StateX.txt", "Q.txt"]
    
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


if __name__ == "__main__":
    main()
