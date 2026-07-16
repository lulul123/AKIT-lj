# 在导入任何可能初始化 OpenMP 的库之前设置环境变量，避免 libiomp5md.dll 重复初始化错误
import os
# 不推荐的临时解决方法：允许重复加载 OpenMP 运行时以避免程序中断（可能有风险）
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")
# 限制线程数以降低不同库间线程冲突的
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
import akit_utils as utils


# 控制输出开关（将不需要的输出设为 False）
SAVE_PRED_PER_EPOCH = False        # 是否保存 predictions_epoch_{n}.csv
SAVE_VALID_PLOT = False            # 是否在每轮保存 验证数据测试.png
GENERATE_TRAIN_PROGRESS = False    # 是否生成 training_progress_epoch_{k}.png（每10轮）
SAVE_DETAILED_ANALYSIS = False     # 是否生成 detailed_analysis.png


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
    
    # 检查数据路径（静默检查，仅在必须时抛出）
    for data_dir in data_dirs:
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"数据路径不存在: {data_dir}")

    # 创建多数据集加载器
    # 合并所有 data_dirs 后按比例划分训练/验证（80/20）
    print("=> 合并数据集并按 80/20 划分训练/验证")
    combined_dataset = MultiDatasetIMU(data_dirs, seq_len=seq_len)
    total_N = len(combined_dataset)
    if total_N == 0:
        raise RuntimeError("No samples found in combined datasets")
    train_N = int(total_N * 0.8)
    val_N = total_N - train_N
    from torch.utils.data import random_split
    train_dataset, val_dataset = random_split(combined_dataset, [train_N, val_N])
    # 打印数据读取相关摘要（保留）
    print(f"Total samples: {total_N}, Train: {train_N}, Val: {val_N}")

    # 可选：不使用单独的 test_loader（若需保留单独测试集，请在外部准备并传入）
    test_dataset = None
    test_loader = None

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

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
    
    # 创建输出目录（不打印路径以减少输出）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"training_results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    # print(f"训练结果将保存到: {output_dir}")
    
    # 训练监控变量
    train_losses = []
    val_losses = []
    epoch_times = []
    batch_losses = []
    learning_rates = []
    q_predictions = []
    position_errors = []
    
    # 在输出目录创建训练日志文件
    test_log_path = os.path.join(output_dir, 'Test.txt')
    with open(test_log_path, 'w', encoding='utf-8') as f:
        f.write('epoch\ttrain_loss\tval_loss\n')
    
    # 5. 训练循环
    # 仅在每个 epoch 打印训练损失
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
            
            # 移除每批次打印，保持输出简洁
            # if (batch_idx + 1) % 10 == 0:
            #     print(...)

        # 记录epoch统计
        avg_loss = total_loss / len(train_loader)
        epoch_time = time.time() - epoch_start_time
        
        train_losses.append(avg_loss)
        epoch_times.append(epoch_time)
        batch_losses.extend(epoch_batch_losses)
        learning_rates.append(optimizer.param_groups[0]['lr'])
        q_predictions.append(np.mean(epoch_q_preds, axis=0))
        position_errors.extend(epoch_pos_errors)

        # 仅打印每个 epoch 的损失（保留）
        # 评估验证集并打印 train/val loss
        model.eval()
        total_val_loss = 0.0
        all_preds = []
        all_trues = []
        with torch.no_grad():
            for vbatch in val_loader:
                v_imu = vbatch["imu"].to(device).double()
                v_pos_true = vbatch["pos_true"].to(device).double()

                v_phi = vbatch["phi"].to(device).double()
                v_H = vbatch["H"].to(device).double()
                v_R = vbatch["R"].to(device).double()
                v_Inno = vbatch["Inno"].to(device).double()
                v_P = vbatch["P"].to(device).double()
                v_stateX = vbatch["stateX"].to(device).double()
                v_Q_origin = vbatch["Q"].to(device).double()

                v_Q_diag = model(v_imu, v_P.unsqueeze(-1))
                v_Q_diag = torch.exp(v_Q_diag)
                v_Q_matrix = torch.diag_embed(v_Q_diag)

                v_pos_predict = kalman_filter(v_phi, v_H, v_R, v_P, v_Inno, v_stateX, v_Q_matrix, v_Q_origin)

                v_loss = torch.mean((v_pos_predict - v_pos_true) ** 2)
                total_val_loss += v_loss.item()
                all_preds.append(v_pos_predict.cpu().numpy())
                all_trues.append(v_pos_true.cpu().numpy())
        avg_val_loss = total_val_loss / max(1, len(val_loader))
        val_losses.append(avg_val_loss)
        # 计算本轮完整耗时（包含训练与验证）并更新记录
        epoch_elapsed = time.time() - epoch_start_time
        if len(epoch_times) > 0:
            epoch_times[-1] = epoch_elapsed
        else:
            epoch_times.append(epoch_elapsed)
        print(f"Epoch [{epoch+1}/{epochs}] train_loss={avg_loss:.6f} val_loss={avg_val_loss:.6f} Elapsed: {epoch_elapsed:.2f}s")
        # 追加到 Test.txt
        with open(test_log_path, 'a', encoding='utf-8') as f:
            f.write(f"{epoch+1}\t{avg_loss:.6f}\t{avg_val_loss:.6f}\n")

        # 保存本轮验证的逐样本预测到 ValidOut.txt（覆盖）和绘制验证图（覆盖）
        if all_preds:
            preds_np = np.concatenate(all_preds, axis=0)
            trues_np = np.concatenate(all_trues, axis=0)
            # 仅在开关开启时写出 ValidOut.txt 与验证图
            if SAVE_PRED_PER_EPOCH or SAVE_VALID_PLOT:
                valid_out_path = os.path.join(output_dir, 'ValidOut.txt')
                with open(valid_out_path, 'w', encoding='utf-8') as vf:
                    for i in range(preds_np.shape[0]):
                        px, py, pz = preds_np[i]
                        tx, ty, tz = trues_np[i]
                        vf.write(f"{i}\t{tx:.6f}\t{px:.6f}\n")
            if SAVE_VALID_PLOT:
                try:
                    import matplotlib
                    matplotlib.use('Agg')
                    import matplotlib.pyplot as plt
                    plt.figure(figsize=(8,4))
                    plt.plot(trues_np[:,0], label='True (X)')
                    plt.plot(preds_np[:,0], label='Pred (X)')
                    plt.legend()
                    plt.title('验证数据测试')
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, '验证数据测试.png'))
                    plt.close()
                except Exception:
                    pass
        
        # 保留周期性图表生成但不打印生成信息
        if GENERATE_TRAIN_PROGRESS and (epoch + 1) % 10 == 0:
            generate_training_plots(train_losses, epoch_times, batch_losses, 
                                   learning_rates, q_predictions, position_errors, 
                                   output_dir, epoch + 1)

    # 训练结束后保存最终时间
    total_training_time = time.time() - start_time
    # 已禁用保存整个模型 model.pt；保留 final_model.pth（state_dict + 信息）
    
    # 生成训练与验证 Loss 曲线并保存
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss')
        if val_losses:
            plt.plot(range(1, len(val_losses)+1), val_losses, label='Valid Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'Loss_vs_Epoch.png'))
        plt.close()
    except Exception:
        pass

    # 在训练结束后，保存训练集的逐样本预测（TrainOut.txt）和训练数据测试图
    try:
        # 生成训练集预测（按顺序）
        ordered_train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
        all_train_preds = []
        all_train_trues = []
        model.eval()
        with torch.no_grad():
            for tb in ordered_train_loader:
                tbatch = tb
                t_imu = tbatch["imu"].to(device).double()
                t_pos_true = tbatch["pos_true"].to(device).double()
                t_P = tbatch["P"].to(device).double()
                t_Q_diag = model(t_imu, t_P.unsqueeze(-1))
                t_Q_diag = torch.exp(t_Q_diag)
                t_Q_matrix = torch.diag_embed(t_Q_diag)
                t_pos_predict = kalman_filter(tbatch["phi"].to(device).double(), tbatch["H"].to(device).double(), tbatch["R"].to(device).double(), tbatch["P"].to(device).double(), tbatch["Inno"].to(device).double(), tbatch["stateX"].to(device).double(), t_Q_matrix, tbatch["Q"].to(device).double())
                all_train_preds.append(t_pos_predict.cpu().numpy())
                all_train_trues.append(t_pos_true.cpu().numpy())
        if all_train_preds:
            train_preds_np = np.concatenate(all_train_preds, axis=0)
            train_trues_np = np.concatenate(all_train_trues, axis=0)
            train_out_path = os.path.join(output_dir, 'TrainOut.txt')
            with open(train_out_path, 'w', encoding='utf-8') as tf:
                for i in range(train_preds_np.shape[0]):
                    # 使用第一个分量作为时间占位
                    tf.write(f"{i}\t{train_trues_np[i,0]:.6f}\t{train_preds_np[i,0]:.6f}\n")
            # 训练数据测试图
            try:
                import matplotlib.pyplot as plt
                plt.figure()
                plt.plot(train_trues_np[:,0], label='True (X)')
                plt.plot(train_preds_np[:,0], label='Pred (X)')
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, '训练数据测试.png'))
                plt.close()
            except Exception:
                pass
    except Exception:
        pass

    # 生成最终报告与图表
    # 仅在需要时生成详细报告与图表
    if SAVE_DETAILED_ANALYSIS:
        generate_final_report(model, train_losses, epoch_times, batch_losses, 
                             learning_rates, q_predictions, position_errors, 
                             output_dir, total_training_time, train_dataset)
    else:
        # 仍然保存基本 final_model.pth 信息
        model_path = os.path.join(output_dir, 'final_model.pth')
        torch.save({
            'model_state_dict': model.state_dict(),
            'train_losses': train_losses,
            'total_time': total_training_time,
            'final_loss': train_losses[-1] if train_losses else 0
        }, model_path)


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
    return utils.load_truePos(file_path)


def load_matrix_file(file_path, rows, cols):
    return utils.load_matrix_file(file_path, rows, cols)


class MultiDatasetIMU(utils.MultiDatasetIMU):
    """包装类，直接继承 akit_utils 中的实现以保持兼容性"""
    pass


class IMUDataset(utils.IMUDataset):
    """包装类，直接继承 akit_utils 中的实现以保持兼容性"""
    pass


def kalman_filter(phi, H, R, P, Inno, stateX, Q_diag, Q_origin):
    return utils.kalman_filter(phi, H, R, P, Inno, stateX, Q_diag, Q_origin)


def check_time_alignment(data_dir):
    return utils.check_time_alignment(data_dir)


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


def run_small_scale_experiment(data_dirs,
                               max_samples=200,
                               epochs=5,
                               batch_size=8,
                               val_ratio=0.1,
                               learning_rate=1e-4,
                               seq_len=100,
                               device=None):
    """
    小规模实验：合并数据后按 max_samples 随机抽取（若指定），再按 80/20 划分为训练/验证
    """
    import csv
    import shutil
    from torch.utils.data import Subset, random_split

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading dataset (small-scale)...")
    full_dataset = MultiDatasetIMU(data_dirs, seq_len=seq_len)
    N = len(full_dataset)
    if N == 0:
        print("没有可用样本，检查数据路径/对齐。")
        return

    use_n = min(max_samples, N)

    # 随机抽取 use_n 个样本（如果 use_n < N）
    if use_n < N:
        indices = np.random.permutation(N)[:use_n].tolist()
        working_dataset = Subset(full_dataset, indices)
    else:
        working_dataset = full_dataset

    # 按 80/20 划分训练/验证
    train_N = int(len(working_dataset) * 0.8)
    val_N = len(working_dataset) - train_N
    if train_N < 1 or val_N < 1:
        raise RuntimeError('Not enough samples for train/val split')

    train_dataset, val_dataset = random_split(working_dataset, [train_N, val_N])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    print(f"Small experiment: total_used={len(working_dataset)}, train={train_N}, val={val_N}")

    # 构建模型
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

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-6)

    # 输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"training_results_small_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Small experiment outputs -> {output_dir}")

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        for batch in train_loader:
            imu = batch["imu"].to(device).double()
            pos_true = batch["pos_true"].to(device).double()

            phi = batch["phi"].to(device).double()
            H = batch["H"].to(device).double()
            R = batch["R"].to(device).double()
            Inno = batch["Inno"].to(device).double()
            P = batch["P"].to(device).double()
            stateX = batch["stateX"].to(device).double()
            Q_origin = batch["Q"].to(device).double()

            Q_diag = model(imu, P.unsqueeze(-1))
            Q_diag = torch.exp(Q_diag)
            Q_matrix = torch.diag_embed(Q_diag)

            pos_predict = kalman_filter(phi, H, R, P, Inno, stateX, Q_matrix, Q_origin)

            loss = torch.mean((pos_predict - pos_true) ** 2)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / max(1, len(train_loader))
        train_losses.append(avg_train_loss)

        # 验证
        model.eval()
        total_val_loss = 0.0
        all_preds = []
        all_trues = []
        with torch.no_grad():
            for vbatch in val_loader:
                v_imu = vbatch["imu"].to(device).double()
                v_pos_true = vbatch["pos_true"].to(device).double()

                v_phi = vbatch["phi"].to(device).double()
                v_H = vbatch["H"].to(device).double()
                v_R = vbatch["R"].to(device).double()
                v_Inno = vbatch["Inno"].to(device).double()
                v_P = vbatch["P"].to(device).double()
                v_stateX = vbatch["stateX"].to(device).double()
                v_Q_origin = vbatch["Q"].to(device).double()

                v_Q_diag = model(v_imu, v_P.unsqueeze(-1))
                v_Q_diag = torch.exp(v_Q_diag)
                v_Q_matrix = torch.diag_embed(v_Q_diag)

                v_pos_predict = kalman_filter(v_phi, v_H, v_R, v_P, v_Inno, v_stateX, v_Q_matrix, v_Q_origin)

                v_loss = torch.mean((v_pos_predict - v_pos_true) ** 2)
                total_val_loss += v_loss.item()

                all_preds.append(v_pos_predict.cpu().numpy())
                all_trues.append(v_pos_true.cpu().numpy())

        avg_val_loss = total_val_loss / max(1, len(val_loader))
        val_losses.append(avg_val_loss)

        # Small-scale: 每轮 predictions 文件保存已禁用
        print(f"[Epoch {epoch+1}/{epochs}] train_loss={avg_train_loss:.6f} val_loss={avg_val_loss:.6f}")

    print(f"Small experiment finished. Train losses: {train_losses}, Val losses: {val_losses}")


# 替换 __main__ 入口以支持 --small 参数
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--small", action="store_true", help="run small-scale experiment")
    args = parser.parse_args()
    if args.small:
        base_path = r"F:\AAA-lulul\科研工作\毕业设计\A-KIT-main\A-KIT-main\data_lj"
        data_dirs = [
            os.path.join(base_path, "data013325"),
            os.path.join(base_path, "data065517"),
            os.path.join(base_path, "data090854"),
            os.path.join(base_path, "data212832"),
            os.path.join(base_path, "data122041")
        ]
        run_small_scale_experiment(data_dirs,
                                   max_samples=200,
                                   epochs=5,
                                   batch_size=8,
                                   val_ratio=0.1,
                                   learning_rate=1e-4,
                                   seq_len=100)
    else:
        main()