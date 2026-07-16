import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
import os
from torch.utils.data import DataLoader
import torch.optim as optim



def main():
    # 1. 训练超参数
    batch_size = 32
    # IMU(t-99 ~ t) → 预测 Q(t)
    seq_len = 100
    # 每个数据集重复训练的轮数
    epochs = 50
    learning_rate = 1e-4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
     
    # 2. 多数据集路径
    data_dirs = [
        "path/to/dataset1",
        "path/to/dataset2", 
        "path/to/dataset3",
        "path/to/dataset4",
        "path/to/dataset5"
    ]
    
    # 创建多数据集加载器
    dataset = MultiDatasetIMU(data_dirs, seq_len=seq_len)
    
    # shuffle=True 打乱时间顺序训练
    # drop_last=True 不足一个 batch 的数据丢弃（保证每个 batch 大小一致）
    train_loader = DataLoader(dataset, batch_size=batch_size,
        shuffle=True, drop_last=True)

    # 3. 构建模型
    # lj:各个参数的含义
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
    )

    model = model.to(device).double()
     
    # 4. 优化器
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-6
    )
     
    # 5. 训练循环
    for epoch in range(epochs):

        model.train()
        total_loss = 0.0

        for batch in train_loader:
             
            # 1 读取输入
            imu = batch["imu"].to(device).double()
            pos_true = batch["pos_true"].to(device).double()
             
            # 2 构造 Kalman 需要的固定矩阵
            # （这些必须提前在 dataset 里准备好）
            phi = batch["phi"].to(device).double()
            H = batch["H"].to(device).double()
            R = batch["R"].to(device).double()
            Inno = batch["Inno"].to(device).double()
            P = batch["P"].to(device).double()
            stateX = batch["stateX"].to(device).double()
            Q_origin = batch["Q"].to(device).double()
             
            # 3. 网络预测 Q 的对角线元素
            # 输出维度: (batch, 15)
            Q_diag = model(imu, P.unsqueeze(-1))

            # 保证 Q 为正（极其重要）
            Q_diag = torch.exp(Q_diag)

            # 构造对角阵 (batch, 15, 15)
            Q_matrix = torch.diag_embed(Q_diag)
             
            # 4. 调用可微 Kalman Filter
            pos_predict = kalman_filter(phi, H, R, P, Inno, stateX, Q_matrix, Q_origin)

             
            # 5. 计算位置 MSE
            loss = torch.mean((pos_predict - pos_true) ** 2)

             
            # 6. 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}]  Loss: {total_loss:.6f}")

    print("Training Finished.")

# 使用示例和时间对齐说明
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


# 时间同步检查函数
def check_time_alignment(data_dir):
    """检查数据时间对齐情况"""
    required_files = ["imu.txt", "Vel.txt", "Inno.txt", "Phi.txt", "H.txt", 
                     "R.txt", "StateP.txt", "StateX.txt", "Q.txt"]
    
    for file in required_files:
        file_path = os.path.join(data_dir, file)
        if not os.path.exists(file_path):
            print(f"Missing file: {file}")
            return False
    
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


# 主函数
if __name__ == "__main__":
    main()


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
        # The input size would be (batch_size,sentence_size,embedding_size)
        batch_size = query.shape[0]
        q, k, v = query.shape[1], key.shape[1], value.shape[1]  # sentence size = how many words

        V = self.Value(value)  # (batch_size,n,embedding_size)
        Q = self.Query(query)  # (batch_size,query_size,embedding_size)
        K = self.Key(key)  # (batch_size,key_size,embedding_size)

        # split to heads for parallel comuptation

        V = V.reshape(batch_size, v, self.num_heads, self.heads_dim)
        V = torch.einsum("bnhd->bhnd", V)  # (batch_size,num_heads,n,embedding_size)
        Q = Q.reshape(batch_size, q, self.num_heads, self.heads_dim)
        Q = torch.einsum("bnhd->bhnd", Q)  # (batch_size,num_heads,n,embedding_size)
        K = K.reshape(batch_size, k, self.num_heads, self.heads_dim)
        K = torch.einsum("bnhd->bhnd", K)  # (batch_size,num_heads,n,embedding_size)

        temp = torch.einsum("bhqd,bhkd->bhqk",
                            [Q, K])  # Q(batch_size,num_heads,n,embedding_size)*K(batch_size,num_heads,embedding_size,n)
        # temp size = (batch_size,num_heads,q,k)

        Attention = torch.softmax(temp / torch.sqrt(torch.tensor(self.embedding_size, dtype=torch.float64)),
                                  dim=3)  # sum of each row is 1
        # Attention size = (batch_size,num_heads,q,k)

        temp = torch.einsum("bhqk,bhkv->bhqv", [Attention, V])
        # temp size = (batch_size,num_heads,n,heads_dim)

        temp = temp.reshape(batch_size, q, self.num_heads * self.heads_dim)
        out = self.fc_out(temp)
        return out


class MAB(nn.Module):
    def __init__(self, embedding_size, num_heads, feedforward_expansion):
        super(MAB, self).__init__()
        self.embedding_size = embedding_size
        self.num_heads = num_heads
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
        self.encoder_layers = nn.ModuleList(
            [
                SAB(embedding_size, num_heads, feedforward_expansion)
                for _ in range(blocklayers)
            ]
        )
        self.decoder_layers = nn.ModuleList(
            [
                Decoder(embedding_size, num_heads, feedforward_expansion, k)
                for _ in range(blocklayers)
            ]
        )
        self.sab = SAB(embedding_size, num_heads, feedforward_expansion)
        self.pma = PMA(embedding_size, num_heads, feedforward_expansion, k)
        self.fc_out = nn.Sequential(
            nn.Linear(embedding_size, feedforward_expansion, dtype=torch.float64),
            nn.LeakyReLU(0.1),
            nn.Linear(feedforward_expansion, output_size, dtype=torch.float64),
        )
        self.fc_out = nn.Linear(embedding_size, output_size, dtype=torch.float64)

    def forward(self, x):
        for layer in self.encoder_layers:
            x = layer(x)
        for layer in self.decoder_layers:
            x = layer(x)
        out = x

        return out


class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, filter_size, in_chans, embed_dim):
        super(PatchEmbedding, self).__init__()
        self.proj = nn.Sequential(
            nn.Conv1d(
                in_chans,
                embed_dim,
                kernel_size=filter_size,
                stride=patch_size, dtype=torch.float64
            )

        )

    def forward(self, x):
        x = self.proj(x)  # (n_samples, embed_dim, n_patches ** 0.5, n_patches ** 0.5)
        x = x.flatten(2)  # (n_samples, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (n_samples, n_patches, embed_dim)

        return x


class Network(nn.Module):
    def __init__(self, patch_size, filter_size, in_chans, embedding_size, num_heads, feedforward_expansion, k,
                 blocklayers, output_size):
        super(Network, self).__init__()
        self.PE = PatchEmbedding(patch_size, filter_size, in_chans, embedding_size)
        self.set_transformer = SetTransformer(embedding_size, num_heads, feedforward_expansion, k, blocklayers,
                                              output_size)
        self.FC_output = nn.Sequential(
            nn.Linear(embedding_size, 32, dtype=torch.float64),
            nn.LeakyReLU(0.1),
            nn.Dropout(p=0.1),  # Dropout layer added
            nn.Linear(32, output_size, dtype=torch.float64),
            nn.ReLU(),
        )
        self.lastlayer = nn.Linear(output_size, output_size, dtype=torch.float64)

    def forward(self, x, Q):
        x = self.PE(x)
        out = self.set_transformer(x)
        out = out.flatten(1)
        out = self.FC_output(out)
        # Create a mask for elements equal to 0
        mask = (out == 0)

        # Replace elements equal to 0 with 1 using the mask
        x = torch.where(mask, torch.tensor(1.0), out)
        x = torch.bmm(torch.diag_embed(x), Q[:, :, :, -2])

        return torch.diagonal(x, dim1=-2, dim2=-1)

# 读取位置真值
def load_truePos(file_path):
    """
    读取包含：
        GPSTime(sec) X-ECEF Y-ECEF Z-ECEF
    返回：
        dict[time] = np.array([X, Y, Z])
    """

    data_dict = {}

    with open(file_path, "r") as f:
        for line in f:

            # 跳过空行
            if not line.strip():
                continue

            # 跳过文件头（包含字母的行）
            if any(c.isalpha() for c in line):
                continue

            values = line.strip().split()

            # 防止异常行
            if len(values) < 4:
                continue

            time = float(values[0])
            x = float(values[1])
            y = float(values[2])
            z = float(values[3])

            data_dict[time] = np.array([x, y, z])

    return data_dict

# 读取矩阵
# 适用于所有的 stateX stateP H R Phi inno
# stateP_dict = load_matrix_file("StateP.txt", 15, 15)  文件名，行数，列数
def load_matrix_file(file_path, rows, cols):
    """
    读取： .txt
    返回： dict[time] = numpy矩阵(rows, cols)
    """
    data_dict = {}

    with open(file_path, "r") as f:
        header = f.readline()  # 跳过标题

        for line in f:
            values = line.strip().split()

            if len(values) < 1 + rows * cols:
                continue  # 防止异常行

            # 第一列是时间
            time = int(values[0])

            # 后面是矩阵元素
            matrix_values = [float(v) for v in values[1:]]

            matrix = np.array(matrix_values).reshape(rows, cols)

            data_dict[time] = matrix

    return data_dict

# 多数据集训练类
class MultiDatasetIMU(Dataset):
    def __init__(self, data_dirs, seq_len=100):
        """
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
        # 1. 文件路径
        imu_file = os.path.join(data_dir, "imu.txt")
        vel_file = os.path.join(data_dir, "Vel.txt")
        
        # 查找位置真值文件（支持不同扩展名）
        pos_files = [f for f in os.listdir(data_dir) 
                    if f.lower().endswith((".imr.ref", ".pos", ".ref"))]
        if not pos_files:
            raise FileNotFoundError(f"No position reference file found in {data_dir}")
        pos_file = os.path.join(data_dir, pos_files[0])
        
        # 矩阵文件路径
        phi_file = os.path.join(data_dir, "Phi.txt")
        H_file = os.path.join(data_dir, "H.txt")
        R_file = os.path.join(data_dir, "R.txt")
        P_file = os.path.join(data_dir, "StateP.txt")
        state_file = os.path.join(data_dir, "StateX.txt")
        inno_file = os.path.join(data_dir, "Inno.txt")
        Q_file = os.path.join(data_dir, "Q.txt")
        
        # 2. 读取高频数据 (IMU + Vel, 100Hz)
        imu_data = self._load_high_freq_data(imu_file, 6)  # 6维IMU
        vel_data = self._load_high_freq_data(vel_file, 3)  # 3维速度
        
        # 3. 时间对齐：找到IMU和Vel的公共时间
        common_times = sorted(set(imu_data.keys()) & set(vel_data.keys()))
        
        # 4. 读取矩阵数据 (1Hz)
        pos_dict = load_truePos(pos_file)
        phi_dict = load_matrix_file(phi_file, 15, 15)
        H_dict = load_matrix_file(H_file, 3, 15)
        R_dict = load_matrix_file(R_file, 3, 3)
        P_dict = load_matrix_file(P_file, 15, 15)
        state_dict = load_matrix_file(state_file, 15, 1)
        inno_dict = load_matrix_file(inno_file, 3, 1)
        Q_dict = load_matrix_file(Q_file, 15, 15)
        
        # 5. 时间对齐：确保矩阵数据和高频数据时间范围一致
        matrix_times = set(pos_dict.keys()) & set(phi_dict.keys()) & set(H_dict.keys()) & \
                      set(R_dict.keys()) & set(P_dict.keys()) & set(state_dict.keys()) & \
                      set(inno_dict.keys()) & set(Q_dict.keys())
        
        # 6. 生成样本
        samples = []
        for i in range(len(common_times) - self.seq_len):
            # 滑动窗口的时间序列
            window_times = common_times[i:i + self.seq_len]
            current_time = window_times[-1]  # 窗口最后时刻
            
            # 对应的矩阵数据时间（取整数秒）
            matrix_time = int(current_time)
            
            # 检查是否有对应的矩阵数据
            if matrix_time not in matrix_times:
                continue
                
            # 构建IMU+Vel窗口数据
            imu_window = []
            for t in window_times:
                if t in imu_data and t in vel_data:
                    combined = imu_data[t] + vel_data[t]  # 6+3=9维
                    imu_window.append(combined)
                else:
                    break
            
            if len(imu_window) != self.seq_len:
                continue
                
            # 创建样本
            sample = {
                "imu": np.array(imu_window, dtype=np.float64),  # (seq_len, 9)
               
                "matrix_time": matrix_time,
                "pos_true": pos_dict[matrix_time],
                "phi": phi_dict[matrix_time],
                "H": H_dict[matrix_time],
                "R": R_dict[matrix_time],
                "P": P_dict[matrix_time],
                "stateX": state_dict[matrix_time],
                "Inno": inno_dict[matrix_time],
                "Q": Q_dict[matrix_time],
            }
            samples.append(sample)
            
        return samples
    
    def _load_high_freq_data(self, file_path, data_dim):
        """
        读取高频数据文件 (IMU/Vel)
        返回: dict[float_time] = [data_values]
        """
        data_dict = {}
        
        with open(file_path, "r") as f:
            header = f.readline()  # 跳过表头
            
            for line in f:
                values = line.strip().split()
                if len(values) < data_dim + 1:
                    continue
                
                # 完整时间戳（包含小数部分）
                time_stamp = float(values[0])
                data_values = [float(v) for v in values[1:data_dim+1]]
                
                data_dict[time_stamp] = data_values
                
        return data_dict

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


# 单数据集类（保持兼容性）
class IMUDataset(Dataset):
    def __init__(self, data_dir, seq_len=100):
        """
        单数据集加载器
        data_dir: 单个数据目录
        seq_len: 滑动窗口长度
        """
        self.multi_dataset = MultiDatasetIMU([data_dir], seq_len)
    
    def __len__(self):
        return len(self.multi_dataset)
    
    def __getitem__(self, idx):
        return self.multi_dataset[idx]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # 滑动时间窗口 窗口长度是seq_len
        imu_window = self.imu[idx:idx + self.seq_len]
        imu_window = torch.tensor(imu_window, dtype=torch.float64)
        # 网络输入要求 (batch, in_chans, seq_len)
        imu_window = imu_window.permute(1, 0)  # 从(seq_len, 9) ->(9, seq_len)
        # 确定当前滤波时刻——窗口的最后一个时间点
        current_time = int(self.time[idx + self.seq_len - 1])

        sample = {
            "imu": imu_window,
            "time": torch.tensor(current_time, dtype=torch.float64),
            "pos_true": torch.tensor(self.pos_dict[current_time], dtype=torch.float64),
            "phi": torch.tensor(self.phi_dict[current_time], dtype=torch.float64),
            "H": torch.tensor(self.H_dict[current_time], dtype=torch.float64),
            "R": torch.tensor(self.R_dict[current_time], dtype=torch.float64),
            "P_prev": torch.tensor(self.P_dict[current_time], dtype=torch.float64),
            "stateX": torch.tensor(self.state_dict[current_time], dtype=torch.float64),
            "Inno": torch.tensor(self.inno_dict[current_time], dtype=torch.float64),
            "Q": torch.tensor(self.Q_dict[current_time], dtype=torch.float64),
        }

        return sample

def kalman_filter(phi, H, R, P, Inno, stateX, Q_diag, Q_origin):
    """
    参数说明:
    phi      : (batch, 15, 15)       状态转移矩阵
    H        : (batch, m, 15)        观测矩阵
    R        : (batch, m, m)         观测噪声
    P        : (batch, 15, 15)       上一时刻协方差
    Inno     : (batch, m, 1)         观测新息
    stateX   : (batch, 15, 1)        上一时刻状态
    Q_diag   : (batch, 15)           过程噪声对角线（网络预测）
    Q_origin : (batch, 15, 15)       原始模型计算的过程噪声
    返回:
    pos_predict : (batch, 3)         预测位置
    """
    # 1️ 状态预测
    x_pred = torch.matmul(phi, stateX)

    # 2. 协方差预测
    phi_T = phi.transpose(-1, -2)
    P_pred = torch.matmul(torch.matmul(phi, P), phi_T) + Q_diag

    # 3. 更新增益矩阵K、误差状态向量X
    H_T = H.transpose(-1, -2)
    # S = H P Hᵀ + R
    S = torch.matmul(torch.matmul(H, P_pred), H_T) + R
    # 数值稳定处理（防止奇异）
    eps = 1e-6
    I_m = torch.eye(S.shape[-1], device=S.device).unsqueeze(0)
    S = S + eps * I_m
    # S⁻¹
    S_inv = torch.linalg.inv(S)
    # K = P Hᵀ S⁻¹
    K = torch.matmul(torch.matmul(P_pred, H_T), S_inv)

    # x_new = x_pred + K * Inno
    x_new = x_pred + torch.matmul(K, Inno)

    # 4. 提取位置预测
    # 假设位置预测在状态前3维
    pos_predict = x_new[:, 0:3, 0]

    return pos_predict


# 使用示例和时间对齐说明
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


# 时间同步检查函数
def check_time_alignment(data_dir):
    """检查数据时间对齐情况"""
    required_files = ["imu.txt", "Vel.txt", "Inno.txt", "Phi.txt", "H.txt", 
                     "R.txt", "StateP.txt", "StateX.txt", "Q.txt"]
    
    for file in required_files:
        file_path = os.path.join(data_dir, file)
        if not os.path.exists(file_path):
            print(f"Missing file: {file}")
            return False
    
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