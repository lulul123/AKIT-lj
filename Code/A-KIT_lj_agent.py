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
import akit_utils_agent as utils


# 控制输出开关（将不需要的输出设为 False）
SAVE_PRED_PER_EPOCH = False        # 是否保存 predictions_epoch_{n}.csv
SAVE_VALID_PLOT = False            # 是否在每轮保存 验证数据测试.png
GENERATE_TRAIN_PROGRESS = False    # 是否生成 training_progress_epoch_{k}.png（每10轮）
SAVE_DETAILED_ANALYSIS = False     # 是否生成 detailed_analysis.png

# 全局快捷放大因子（用于诊断：将模型输出的 Q 对角线放大以提升梯度信号）
SCALE_Q = 1.0


def main():
    # ====== 新增：随机种子 ======
    torch.manual_seed(42)
    np.random.seed(42)

    # 1. 训练超参数
    batch_size = 32
    seq_len = 100  # IMU(t-99 ~ t) → 预测 Q(t)
    epochs = 50
    learning_rate = 5e-3

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

    # 计算训练集全局 IMU per-channel 均值/方差（用于统一归一化）
    try:
        imu_sum = None
        imu_sumsq = None
        imu_count = 0
        CHANNELS = 9
        for idx in range(len(train_dataset)):
            sample = train_dataset[idx]
            s_imu = sample["imu"]
            if torch.is_tensor(s_imu):
                s_imu = s_imu.cpu().numpy()
            # 期望 s_imu 为 (C, L) 或 (L, C)
            if s_imu.ndim == 2:
                if s_imu.shape[0] == CHANNELS:
                    arr = s_imu  # (C, L)
                elif s_imu.shape[1] == CHANNELS:
                    arr = s_imu.T
                else:
                    arr = s_imu.reshape(CHANNELS, -1)
            else:
                arr = s_imu.reshape(CHANNELS, -1)
            # 按通道累加
            if imu_sum is None:
                imu_sum = np.sum(arr, axis=1)
                imu_sumsq = np.sum(arr * arr, axis=1)
            else:
                imu_sum += np.sum(arr, axis=1)
                imu_sumsq += np.sum(arr * arr, axis=1)
            imu_count += arr.shape[1]
        if imu_count > 0:
            imu_mean = imu_sum / imu_count
            imu_var = imu_sumsq / imu_count - imu_mean ** 2
            imu_std = np.sqrt(np.maximum(imu_var, 1e-8))
        else:
            imu_mean = np.zeros((CHANNELS,), dtype=float)
            imu_std = np.ones((CHANNELS,), dtype=float)
    except Exception:
        imu_mean = np.zeros((9,), dtype=float)
        imu_std = np.ones((9,), dtype=float)

    # 转换为 tensor，便于训练时广播，形状 (1, C, 1)（使用 float32 提高性能）
    # 使用 double 精度以保留卡尔曼更新的小量增量
    imu_mean_tensor = torch.tensor(imu_mean, dtype=torch.float64).view(1, -1, 1).to(device)
    imu_std_tensor = torch.tensor(imu_std, dtype=torch.float64).view(1, -1, 1).to(device)

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    # ========== 快速自检配置 ==========
    # 若为 True，训练前会对第一个训练批次运行 single-batch overfit 来拟合 Q_origin 的对角线
    RUN_OVERFIT_MAIN = True
    OVERFIT_STEPS = 200
    OVERFIT_LR = 1e-3
    # ========== 结束配置 ==========

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
    # 确保所有模型参数为 double
    model = model.double()

    # 快速自检：打印第一个训练批次的 Q_origin 对角线统计与 pos_true 量级，并可运行 single-batch overfit
    try:
        sample_batch = next(iter(train_loader))
        sb_Q_origin = sample_batch["Q"].to(device).float()
        try:
            sb_Q_diag = sb_Q_origin.diagonal(offset=0, dim1=1, dim2=2)
        except Exception:
            sb_Q_diag = torch.stack([torch.diagonal(m) for m in sb_Q_origin], dim=0)
        with torch.no_grad():
            qmin = float(sb_Q_diag.min().cpu().numpy())
            qmax = float(sb_Q_diag.max().cpu().numpy())
            qmean = float(sb_Q_diag.mean().cpu().numpy())
        print(f"[QC] Q_origin diag: min={qmin:.6e}, mean={qmean:.6e}, max={qmax:.6e}")
        # pos_true 量级
        sb_pos = sample_batch.get("pos_true", None)
        if sb_pos is not None:
            try:
                pmin = float(sb_pos.min().cpu().numpy())
                pmax = float(sb_pos.max().cpu().numpy())
                pmean = float(sb_pos.mean().cpu().numpy())
                print(f"[QC] pos_true: min={pmin:.6e}, mean={pmean:.6e}, max={pmax:.6e}")
            except Exception:
                pass

        # single-batch overfit 测试（拟合 Q_origin 对角线），只拟合模型输出到 target_diag（不乘 SCALE_Q）
        if RUN_OVERFIT_MAIN:
            print("[QC] Running single-batch overfit to Q_origin diagonal (steps=", OVERFIT_STEPS, ")...")
            model.train()
            o_imu = sample_batch["imu"].to(device).float()
            o_P = sample_batch["P"].to(device).float()
            o_Q_origin = sb_Q_origin
            try:
                target_diag = o_Q_origin.diagonal(offset=0, dim1=1, dim2=2)
            except Exception:
                target_diag = torch.stack([torch.diagonal(m) for m in o_Q_origin], dim=0)
            overfit_optimizer = optim.Adam(model.parameters(), lr=OVERFIT_LR)
            last_loss = None
            for step in range(1, OVERFIT_STEPS + 1):
                overfit_optimizer.zero_grad()
                out_raw = model(o_imu, o_P.unsqueeze(-1))
                out_pred = torch.nn.functional.softplus(out_raw)
                loss_of = torch.mean((out_pred - target_diag) ** 2)
                loss_of.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                overfit_optimizer.step()
                last_loss = loss_of.item()
                if step % max(1, OVERFIT_STEPS // 5) == 0 or step <= 5:
                    print(f"[QC overfit] step={step}/{OVERFIT_STEPS} loss={last_loss:.6e}")
            print(f"[QC overfit] finished. final loss={last_loss:.6e}")
    except StopIteration:
        print("[QC] Train loader empty, skipping quick checks.")
    except Exception as e:
        print("[QC] Quick-check failed:", e)

    # 4. 优化器
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-6)

    # ====== Diagnostic: verify parameters registered in optimizer and require_grad ======
    try:
        params_list = list(model.parameters())
        print('\n[Diag] Model parameters (first 3):')
        for i, p in enumerate(params_list[:3]):
            print(f" [Diag] Param[{i}] shape={tuple(p.shape)} requires_grad={p.requires_grad}")
        # Collect ids of params in optimizer
        opt_param_ids = set()
        for g in optimizer.param_groups:
            for pp in g.get('params', []):
                opt_param_ids.add(id(pp))
        for i, p in enumerate(params_list[:3]):
            in_opt = id(p) in opt_param_ids
            print(f" [Diag] Param[{i}] registered in optimizer: {in_opt}")
    except Exception as _:
        pass
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )

    # 创建输出目录（不打印路径以减少输出）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"training_results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # 新增：创建每批/每轮监控文件
    batch_metrics_path = os.path.join(output_dir, 'batch_metrics.csv')
    epoch_metrics_path = os.path.join(output_dir, 'epoch_metrics.csv')
    try:
        with open(batch_metrics_path, 'w', encoding='utf-8') as bf:
            bf.write('epoch,batch,train_loss,Q_min,Q_mean,Q_max,grad_Q_norm,param_grad_norm,pos_error_mean\n')
        with open(epoch_metrics_path, 'w', encoding='utf-8') as ef:
            ef.write('epoch,train_loss,val_loss,avg_Q_mean,avg_pos_error\n')
    except Exception:
        pass

    # 保存统计以便复现（确保 output_dir 已创建）
    try:
        np.save(os.path.join(output_dir, 'imu_mean.npy'), imu_mean)
        np.save(os.path.join(output_dir, 'imu_std.npy'), imu_std)
    except Exception:
        pass
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
    
    # ====== 新增：Early Stopping ======
    best_val_loss = float('inf')
    patience = 10
    early_stop_counter = 0


    for epoch in range(epochs):
        epoch_start_time = time.time()
        model.train()
        total_loss = 0.0
        epoch_batch_losses = []
        epoch_pos_errors = []
        epoch_q_preds = []

        for batch_idx, batch in enumerate(train_loader):
            # 读取输入
            imu = batch["imu"].to(device).float()
            pos_true = batch["pos_true"].to(device).float()
             

            # ====== 新增：归一化 ======
            # 使用训练集统计量做归一化
            try:
                imu = (imu - imu_mean_tensor) / imu_std_tensor
            except Exception:
                pass

            # --- 新增：输入分布实时监控 ---
            if batch_idx % 20 == 0:
                try:
                    with torch.no_grad():
                        imu_min = float(imu.min().cpu().numpy())
                        imu_max = float(imu.max().cpu().numpy())
                        imu_mean_b = float(imu.mean().cpu().numpy())
                        pos_min = float(pos_true.min().cpu().numpy())
                        pos_max = float(pos_true.max().cpu().numpy())
                    print(f"Batch {batch_idx}/{len(train_loader)} | imu(min,max,mean) = ({imu_min:.3e},{imu_max:.3e},{imu_mean_b:.3e}) | pos(min,max) = ({pos_min:.3e},{pos_max:.3e})")
                except Exception:
                    pass

            # 构造 Kalman 需要的矩阵
            phi = batch["phi"].to(device).float()
            H = batch["H"].to(device).float()
            R = batch["R"].to(device).float()
            Inno = batch["Inno"].to(device).float()
            P = batch["P"].to(device).float()
            stateX = batch["stateX"].to(device).float()
            Q_origin = batch["Q"].to(device).float()
             

            # 网络预测 Q 的对角线元素
            # 1. 网络预测 Q 的对角线元素 (此时是 Float)
            Q_diag_float = model(imu.to(torch.float32), P.unsqueeze(-1).to(torch.float32))
            
            # 使用 softplus 保证为正且更稳定，但添加偏置避免太小
            Q_diag_float = torch.nn.functional.softplus(Q_diag_float) + 1.0  # 添加偏置
            
            # 2. 【关键修改】显式转为 Double 以便进入卡尔曼滤波
            # 这样在 backward 时，PyTorch 会自动处理从 Double 到 Float 的梯度类型转换
            # 增大缩放因子以增强梯度信号
            Q_diag = Q_diag_float.to(torch.float64) * SCALE_Q

            # --- DEBUG: 强制检查计算图连通性（在调用 kalman_filter 之前） ---
            try:
                print("DEBUG: Q_diag.requires_grad=", Q_diag.requires_grad, "| Q_diag.grad_fn=", Q_diag.grad_fn)
                params_dbg = list(model.parameters())[:3]
                for i, p in enumerate(params_dbg):
                    print(f"DEBUG: Param[{i}] requires_grad={p.requires_grad} shape={tuple(p.shape)}")
            except Exception:
                pass
            
            # --- 实时监控 Q 分布（每批） ---
            try:
                with torch.no_grad():
                    q_min = Q_diag.min().item()
                    q_max = Q_diag.max().item()
                    q_mean = Q_diag.mean().item()
                print(f"Batch [{batch_idx}/{len(train_loader)}] | Q_min: {q_min:.8e} | Q_max: {q_max:.8e} | Q_mean: {q_mean:.8e}")
            except Exception:
                q_min = None
                q_max = None
                q_mean = None

            # --- 检查 NaN 预警 ---
            if torch.isnan(Q_diag).any():
                print("🧨 DETECTED NaN in Q_diag! Training stopped.")
                break
            
            # 直接传入 Q_diag (B,n) 由 kalman_filter 内部处理为对角缩放或按元素矩阵
            pos_predict = kalman_filter(phi, H, R, P, Inno, stateX, Q_diag, Q_origin)
            
            # --- 诊断：检查 pos_predict 是否与 Q_diag 相关 ---
            # 只在第一个batch检查
            if batch_idx == 0:
                try:
                    with torch.no_grad():
                        # 检查是否可以通过 grad_fn 追踪到 Q_diag
                        if pos_predict.grad_fn is not None:
                            print(f"DEBUG: pos_predict has grad_fn, computation graph is connected")
                        else:
                            print("WARNING: pos_predict has NO grad_fn! Graph may be broken")
                except Exception:
                    pass
             
            # --- 计算 Loss (直接位置误差) ---
            # 明确取前 3 列，并确保是 float64
            pos_true_xyz = pos_true[:, :3].to(torch.float64) 
            
            # 直接计算位置误差损失
            loss = nn.MSELoss()(pos_predict, pos_true_xyz)
            # -----------------------
            
            # 计算位置误差（米）
            pos_error = torch.sqrt(torch.mean((pos_predict - pos_true) ** 2, dim=1))
            
            # 反向传播（真正更新梯度）
            optimizer.zero_grad()
            loss.backward()
            
            # --- 梯度监控：检查关键参数的梯度 ---
            if batch_idx % 10 == 0:
                try:
                    # 检查网络输出的梯度
                    if Q_diag_float.grad is not None:
                        grad_norm_q = torch.norm(Q_diag_float.grad).item()
                        print(f"Grad monitor: Q_diag_float grad norm = {grad_norm_q:.6e}")
                    
                    # 检查模型参数的梯度
                    total_grad_norm = 0.0
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            param_grad_norm = torch.norm(param.grad).item()
                            total_grad_norm += param_grad_norm
                            if "weight" in name and "FC_output" in name:
                                print(f"Grad monitor: {name} grad norm = {param_grad_norm:.6e}")
                    print(f"Grad monitor: Total gradient norm = {total_grad_norm:.6e}")
                except Exception as e:
                    print(f"Grad monitor error: {e}")

            # --- 3. 梯度裁剪与监控 (合并为一个操作，提高效率) ---
            # clip_grad_norm_ 会返回裁剪前的总梯度范数，直接利用它做监控，不要自己写循环算
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            total_norm = float(total_norm.item())   
            optimizer.step()

            # --- 4. 必要的诊断打印 (建议每 10 个 batch 打印一次) ---
            if batch_idx % 10 == 0:
                # 只保留最核心的指标
                pos_error_val = torch.sqrt(torch.mean((pos_predict - pos_true) ** 2)).item()
                print(f"Batch {batch_idx} | Loss: {loss.item():.6e} | GradNorm: {total_norm:.4e} | PosErr: {pos_error_val:.3f}m")
                
                if total_norm == 0.0:
                    print(f"⚠️ Warning: zero gradient at batch {batch_idx}")

            # --- 5. 写入 CSV 监控 (保留必要的指标) ---
            try:
                # 尽量在一次写操作中完成
                with open(batch_metrics_path, 'a', encoding='utf-8') as bf:
                    # 获取 Q 的统计信息 (如果需要监控网络输出的变化)
                    q_m = Q_diag.mean().item()
                    bf.write(f"{epoch+1},{batch_idx+1},{loss.item():.6e},{q_m:.6e},{total_norm:.6e}\n")
            except Exception:
                pass

            # 记录数据
            batch_loss = loss.item()
            total_loss += batch_loss
            epoch_batch_losses.append(batch_loss)
            epoch_pos_errors.extend(pos_error.cpu().detach().numpy())
            epoch_q_preds.append(Q_diag[0].cpu().detach().numpy())  # 记录第一个样本的Q预测


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
                v_imu = vbatch["imu"].to(device).float()
                v_pos_true = vbatch["pos_true"].to(device).float()

                # ====== 新增：归一化（用当前batch统计）=====
                # 使用训练集统计量对验证数据归一化
                try:
                    v_imu = (v_imu - imu_mean_tensor) / imu_std_tensor
                except Exception:
                    pass

                v_phi = vbatch["phi"].to(device).float()
                v_H = vbatch["H"].to(device).float()
                v_R = vbatch["R"].to(device).float()
                v_Inno = vbatch["Inno"].to(device).float()
                v_P = vbatch["P"].to(device).float()
                v_stateX = vbatch["stateX"].to(device).float()
                v_Q_origin = vbatch["Q"].to(device).float()

                v_Q_diag_float = model(v_imu.to(torch.float32), v_P.unsqueeze(-1).to(torch.float32))
                v_Q_diag_float = torch.nn.functional.softplus(v_Q_diag_float) + 1.0
                v_Q_diag = v_Q_diag_float.to(torch.float64) * SCALE_Q
                # 直接传入 v_Q_diag
                v_pos_predict = kalman_filter(v_phi, v_H, v_R, v_P, v_Inno, v_stateX, v_Q_diag, v_Q_origin)

                v_loss = torch.mean((v_pos_predict - v_pos_true) ** 2)
                total_val_loss += v_loss.item()
                all_preds.append(v_pos_predict.cpu().numpy())
                all_trues.append(v_pos_true.cpu().numpy())
        avg_val_loss = total_val_loss / max(1, len(val_loader))
        val_losses.append(avg_val_loss)

        # ====== 新增：每 epoch 写入 epoch_metrics.csv ======
        try:
            avg_Q_mean = np.mean(q_predictions[-1]) if q_predictions else ''
            avg_pos_err = np.mean(epoch_pos_errors) if epoch_pos_errors else ''
            with open(epoch_metrics_path, 'a', encoding='utf-8') as ef:
                ef.write(f"{epoch+1},{avg_loss:.6e},{avg_val_loss:.6e},{avg_Q_mean},{avg_pos_err}\n")
        except Exception:
            pass

        # 仅打印每个 epoch 的损失
        print(f"Epoch [{epoch+1}/{epochs}] train_loss={avg_loss:.6f} val_loss={avg_val_loss:.6f} Elapsed: {epoch_time:.2f}s")
        
        # ====== 新增：更新学习率 ======
        scheduler.step(avg_val_loss)

        # ====== 新增：Early Stopping判断 ======
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0

            # 保存最优模型
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pth"))
        else:
            early_stop_counter += 1

        if early_stop_counter >= patience:
            print("Early stopping triggered!")
            break
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
                t_imu = tbatch["imu"].to(device).float()
                t_pos_true = tbatch["pos_true"].to(device).float()
                t_P = tbatch["P"].to(device).float()

                # ====== 新增：应用训练时相同的归一化到 TrainOut 阶段 ======
                try:
                    t_imu = (t_imu - imu_mean_tensor) / imu_std_tensor
                except Exception:
                    pass

                t_Q_diag = model(t_imu, t_P.unsqueeze(-1))
                t_Q_diag = torch.nn.functional.softplus(t_Q_diag)
                # 放大用于 TrainOut 的 Q_diag
                t_Q_diag = t_Q_diag * SCALE_Q
                t_pos_predict = kalman_filter(tbatch["phi"].to(device).float(), tbatch["H"].to(device).float(), tbatch["R"].to(device).float(), tbatch["P"].to(device).float(), tbatch["Inno"].to(device).float(), tbatch["stateX"].to(device).float(), t_Q_diag, tbatch["Q"].to(device).float())
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

        self.Query = nn.Linear(embedding_size, embedding_size, bias=False, dtype=torch.float32)
        self.Key = nn.Linear(embedding_size, embedding_size, bias=False, dtype=torch.float32)
        self.Value = nn.Linear(embedding_size, embedding_size, bias=False, dtype=torch.float32)
        self.fc_out = nn.Linear(embedding_size, embedding_size, bias=False, dtype=torch.float32)

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
        Attention = torch.softmax(temp / torch.sqrt(torch.tensor(self.embedding_size, dtype=torch.float32)), dim=3)
        temp = torch.einsum("bhqk,bhkv->bhqv", [Attention, V])

        temp = temp.reshape(batch_size, q, self.num_heads * self.heads_dim)
        out = self.fc_out(temp)
        return out


class MAB(nn.Module):
    def __init__(self, embedding_size, num_heads, feedforward_expansion):
        super(MAB, self).__init__()
        self.MHA = MultiHeadAttention(embedding_size, num_heads)
        self.lnorm1 = nn.LayerNorm(embedding_size, dtype=torch.float32)
        self.lnorm2 = nn.LayerNorm(embedding_size, dtype=torch.float32)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_size, feedforward_expansion, dtype=torch.float32),
            nn.LeakyReLU(0.1),
            nn.Linear(feedforward_expansion, embedding_size, dtype=torch.float32),
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
            torch.nn.Parameter(torch.zeros(k, embedding_size, dtype=torch.float32), requires_grad=True))
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_size, feedforward_expansion, dtype=torch.float32),
            nn.LeakyReLU(0.1),
            nn.Linear(feedforward_expansion, embedding_size, dtype=torch.float32),
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
        self.fc_out = nn.Linear(embedding_size, output_size, dtype=torch.float32)

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
                             stride=patch_size, dtype=torch.float32)

    def forward(self, x):
        # 强制输入与卷积权重 dtype 一致，避免 float/double mismatch
        x = x.to(dtype=self.proj.weight.dtype, device=x.device)
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
            nn.Linear(embedding_size, 32, dtype=torch.float32),
            nn.LeakyReLU(0.1),
            nn.Dropout(p=0.1),
            nn.Linear(32, output_size, dtype=torch.float32),
        )

    def forward(self, x, Q):
        x = self.PE(x)
        out = self.set_transformer(x)
        out = out.flatten(1)
        out = self.FC_output(out)
        
        # 确保输出形状是 (batch, 15)，对应15维状态的对角缩放因子
        # 如果 output_size 不是15，需要适配
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
                               learning_rate=1e-2,
                               seq_len=100,
                               device=None,
                               run_overfit=False,
                               overfit_steps=500,
                               overfit_lr=1e-3):
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

    # 计算训练集全局 IMU per-channel 均值/方差（用于统一归一化）
    try:
        imu_sum = None
        imu_sumsq = None
        imu_count = 0
        CHANNELS = 9
        for idx in range(len(train_dataset)):
            sample = train_dataset[idx]
            s_imu = sample["imu"]
            if torch.is_tensor(s_imu):
                s_imu = s_imu.cpu().numpy()
            # 期望 s_imu 为 (C, L) 或 (L, C)
            if s_imu.ndim == 2:
                if s_imu.shape[0] == CHANNELS:
                    arr = s_imu  # (C, L)
                elif s_imu.shape[1] == CHANNELS:
                    arr = s_imu.T
                else:
                    arr = s_imu.reshape(CHANNELS, -1)
            else:
                arr = s_imu.reshape(CHANNELS, -1)
            # 按通道累加
            if imu_sum is None:
                imu_sum = np.sum(arr, axis=1)
                imu_sumsq = np.sum(arr * arr, axis=1)
            else:
                imu_sum += np.sum(arr, axis=1)
                imu_sumsq += np.sum(arr * arr, axis=1)
            imu_count += arr.shape[1]
        if imu_count > 0:
            imu_mean = imu_sum / imu_count
            imu_var = imu_sumsq / imu_count - imu_mean ** 2
            imu_std = np.sqrt(np.maximum(imu_var, 1e-8))
        else:
            imu_mean = np.zeros((CHANNELS,), dtype=float)
            imu_std = np.ones((CHANNELS,), dtype=float)
    except Exception:
        imu_mean = np.zeros((9,), dtype=float)
        imu_std = np.ones((9,), dtype=float)

    # 转换为 tensor，便于训练时广播，形状 (1, C, 1)（使用 float32 提高性能）
    # 使用 double 精度以保留卡尔曼更新的小量增量
    imu_mean_tensor = torch.tensor(imu_mean, dtype=torch.float64).view(1, -1, 1).to(device)
    imu_std_tensor = torch.tensor(imu_std, dtype=torch.float64).view(1, -1, 1).to(device)

    # 创建输出目录（不打印路径以减少输出）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"training_results_small_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Small experiment outputs -> {output_dir}")

    # 构建模型与优化器（与 main 保持一致，使用 float32）
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
    ).to(device)
    # 在小实验中也使用 double
    model = model.double()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-6)

    # ===== 单批过拟合测试（默认关闭） =====
    if run_overfit:
        print("Running single-batch overfit test...")
        try:
            small_iter = iter(train_loader)
            obatch = next(small_iter)
        except Exception:
            print("No batch available for overfit test")
            obatch = None
        if obatch is not None:
            o_imu = obatch["imu"].to(device).float()
            o_P = obatch["P"].to(device).float()
            o_Q_origin = obatch["Q"].to(device).float()
            
            # 目标：使用 Q_origin 的对角线作为网络目标，形状 (B, D)
            try:
                target_diag = o_Q_origin.diagonal(offset=0, dim1=1, dim2=2)
            except Exception:
                target_diag = torch.stack([torch.diagonal(m) for m in o_Q_origin], dim=0)
            
            overfit_optimizer = optim.Adam(model.parameters(), lr=overfit_lr)
            model.train()
            for step in range(1, overfit_steps + 1):
                overfit_optimizer.zero_grad()
                out_raw = model(o_imu, o_P.unsqueeze(-1))
                out_pred = torch.nn.functional.softplus(out_raw)
                loss_of = torch.mean((out_pred - target_diag) ** 2)
                loss_of.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                overfit_optimizer.step()
                if step % max(1, overfit_steps // 10) == 0 or step <= 5:
                    # 打印当前 step 的 loss 与梯度范数
                    try:
                        total_norm_sq_tensor = torch.tensor(0.0, device=device, dtype=torch.float32)
                        max_g_tensor = torch.tensor(0.0, device=device, dtype=torch.float32)
                        for p in model.parameters():
                            if p.grad is not None:
                                g = p.grad.data.norm(2)
                                total_norm_sq_tensor += g * g
                                max_g_tensor = torch.maximum(max_g_tensor, g)
                        total_norm = float(torch.sqrt(total_norm_sq_tensor).item())
                        max_g = float(max_g_tensor.item())
                    except Exception:
                        total_norm = 0.0
                        max_g = 0.0
                    print(f"[overfit] step={step}/{overfit_steps} loss={loss_of.item():.6e} grad_norm={total_norm:.6e} grad_max={max_g:.6e}")
        print("Single-batch overfit test finished.")
    
    # 保存统计以便复现（确保 output_dir 已创建）
    try:
        np.save(os.path.join(output_dir, 'imu_mean.npy'), imu_mean)
        np.save(os.path.join(output_dir, 'imu_std.npy'), imu_std)
    except Exception:
        pass

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):
            imu = batch["imu"].to(device).float()
            pos_true = batch["pos_true"].to(device).float()

            phi = batch["phi"].to(device).float()
            H = batch["H"].to(device).float()
            R = batch["R"].to(device).float()
            Inno = batch["Inno"].to(device).float()
            P = batch["P"].to(device).float()
            stateX = batch["stateX"].to(device).float()
            Q_origin = batch["Q"].to(device).float()
             

            # 网络预测 Q 的对角线元素
            Q_diag = model(imu, P.unsqueeze(-1))
            Q_diag = torch.nn.functional.softplus(Q_diag)
            # 小实验训练阶段放大
            Q_diag = Q_diag * SCALE_Q

            # --- 小实验：每批监控 Q 分布（min/max/mean） ---
            try:
                with torch.no_grad():
                    q_min = Q_diag.min().item()
                    q_max = Q_diag.max().item()
                    q_mean = Q_diag.mean().item()
                print(f"[small] Batch [{batch_idx}/{len(train_loader)}] | Q_min: {q_min:.8e} | Q_max: {q_max:.8e} | Q_mean: {q_mean:.8e}")
            except Exception:
                pass

            pos_predict = kalman_filter(phi, H, R, P, Inno, stateX, Q_diag, Q_origin)

            loss = torch.mean((pos_predict - pos_true) ** 2)
            
            # ====== 小实验：在 backward 之前做 autograd.grad 检查 ======
            try:
                try:
                    params_check = list(model.parameters())[:3]
                    grads = torch.autograd.grad(loss, [Q_diag] + params_check, retain_graph=True, allow_unused=True)
                    gQ = grads[0]
                    gparams = grads[1:]
                    if gQ is None:
                        print('[small] pre-backward: grad(loss, Q_diag) is None')
                    else:
                        print('[small] pre-backward: grad(loss, Q_diag) norm:', float(gQ.norm()))
                    for i, gg in enumerate(gparams):
                        if gg is None:
                            print(f'[small] pre-backward: param[{i}] grad is None')
                        else:
                            print(f'[small] pre-backward: param[{i}] grad norm:', float(gg.norm()))
                except Exception as e:
                    print('[small] pre-backward autograd.grad failed:', e)
            except Exception:
                pass

            optimizer.zero_grad()
            loss.backward()

            # --- 小实验：每批监控梯度范数（总范数与最大单参数梯度） ---
            try:
                # 在设备上累加，避免对每个参数调用 .item()（会导致设备同步/阻塞）
                total_norm_sq_tensor = torch.tensor(0.0, device=device, dtype=torch.float32)
                max_grad_tensor = torch.tensor(0.0, device=device, dtype=torch.float32)
                for p in model.parameters():
                    if p.grad is not None:
                        g = p.grad.data.norm(2)
                        total_norm_sq_tensor += g * g
                        max_grad_tensor = torch.maximum(max_grad_tensor, g)
                total_norm = float(torch.sqrt(total_norm_sq_tensor).item())
                max_grad = float(max_grad_tensor.item())
                print(f"[small] Grad norm: {total_norm:.6e} | Grad max: {max_grad:.6e}")
                if total_norm == 0.0:
                    print("[small] ⚠️ Warning: zero gradient detected")
                if total_norm > 1e6:
                    print("[small] ⚠️ Warning: very large gradient norm", total_norm)
            except Exception:
                total_norm = 0.0

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
                v_imu = vbatch["imu"].to(device).float()
                v_pos_true = vbatch["pos_true"].to(device).float()

                v_phi = vbatch["phi"].to(device).float()
                v_H = vbatch["H"].to(device).float()
                v_R = vbatch["R"].to(device).float()
                v_Inno = vbatch["Inno"].to(device).float()
                v_P = vbatch["P"].to(device).float()
                v_stateX = vbatch["stateX"].to(device).float()
                v_Q_origin = vbatch["Q"].to(device).float()

                v_Q_diag = model(v_imu, v_P.unsqueeze(-1))
                v_Q_diag = torch.nn.functional.softplus(v_Q_diag)
                # 小实验验证阶段放大
                v_Q_diag = v_Q_diag * SCALE_Q
                v_pos_predict = kalman_filter(v_phi, v_H, v_R, v_P, v_Inno, v_stateX, v_Q_diag, v_Q_origin)

                v_loss = torch.mean((v_pos_predict - v_pos_true) ** 2)
                total_val_loss += v_loss.item()

                all_preds.append(v_pos_predict.cpu().numpy())
                all_trues.append(v_pos_true.cpu().numpy())
        avg_val_loss = total_val_loss / max(1, len(val_loader))
        val_losses.append(avg_val_loss)

        print(f"Epoch [{epoch+1}/{epochs}] train_loss={avg_train_loss:.6f} val_loss={avg_val_loss:.6f}")

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
                                   seq_len=100,
                                   run_overfit=True)
    else:
        main()