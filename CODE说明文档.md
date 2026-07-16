# Code 目录 Python 代码说明文档

## 1. 文档范围与结论摘要

本文档覆盖 `Code` 目录下除 `__pycache__` 外的全部 8 个 Python 文件，共约 6091 行：

| 文件 | 行数 | 定位 | 是否有命令行入口 |
|---|---:|---|---|
| `A-KIT.py` | 732 | 最早期的训练原型，包含模型、数据集和 Kalman 更新 | 有，但入口位置导致直接运行失败 |
| `A-KIT_clean.py` | 563 | 对早期原型的整理版 | 有 |
| `A-KIT_copy.py` | 1181 | 加入当前参考文件/多行矩阵解析、IMU/速度匹配、训练报告与调试输出的版本 | 有 |
| `A-KIT_lj.py` | 883 | 将数据与 Kalman 逻辑抽到工具模块，并加入训练/验证划分、小规模实验和输出开关 | 有，支持 `--small` |
| `A-KIT_lj_agent.py` | 1353 | 当前诊断增强版；加入归一化、梯度诊断、学习率调度、早停和单批过拟合自检 | 有，支持 `--small` |
| `akit_utils_agent.py` | 511 | `A-KIT_lj_agent.py` 使用的共享数据读取、数据集和可微 Kalman 工具 | 无 |
| `data_test.py` | 526 | 独立的数据读取与张量形状检查脚本 | 有 |
| `resample_500_to_100.py` | 342 | 高频 IMU/速度文本数据的 500 Hz → 100 Hz 下采样工具 | 有，使用 `argparse` |

整体目标是：将每个整数秒内的 100 个 IMU 六维样本与速度三维样本拼成 `(9, 100)` 输入，通过卷积分块和 Set Transformer 预测 15 维过程噪声缩放量，再把该缩放量送入单步可微 Kalman 更新，以更新后状态的前三维位置与参考位置之间的均方误差训练网络。

需要特别注意：仓库同时保留了多代实验脚本，它们并不是同一条都可直接运行的生产流水线。按当前文件依赖看，`A-KIT_lj_agent.py` + `akit_utils_agent.py` 是依赖闭合的一组；`A-KIT_lj.py` 导入的 `akit_utils.py` 在当前仓库中不存在。各训练入口又写死了 Windows 数据绝对路径，因此在当前 Linux 仓库路径下不能不经配置直接完成训练。哪个版本应被认定为正式入口，代码中没有声明，**需要进一步确认**。

## 2. 文件间依赖关系

### 2.1 仓库内依赖

```text
                           +------------------------+
                           |  akit_utils_agent.py   |
                           |  数据读取 / Dataset /   |
                           |  differentiable KF     |
                           +-----------^------------+
                                       |
                                       | import as utils；包装/转发
                           +-----------+------------+
                           | A-KIT_lj_agent.py      |
                           | 诊断增强训练入口        |
                           +------------------------+

 +------------------+      import akit_utils       +----------------------+
 | A-KIT_lj.py      | ----------------------------> | 当前仓库中不存在     |
 | 训练/验证版本     |                               | akit_utils.py         |
 +------------------+                               +----------------------+

 +----------+  +----------------+  +----------------+  +--------------+
 | A-KIT.py |  | A-KIT_clean.py |  | A-KIT_copy.py |  | data_test.py |
 +----------+  +----------------+  +----------------+  +--------------+
      各自内置模型/数据读取/Kalman，彼此没有 Python import 依赖

 +------------------------+
 | resample_500_to_100.py |
 +------------------------+
      独立预处理命令行工具，不被其他 Python 文件导入
```

`A-KIT_lj.py` 和 `A-KIT_lj_agent.py` 虽然仍定义同名的 `load_truePos`、`load_matrix_file`、`MultiDatasetIMU`、`IMUDataset`、`kalman_filter`、`check_time_alignment`，但这些只是对各自 `utils` 模块的转发函数或继承包装类。

### 2.2 外部依赖

| 依赖 | 用途 | 涉及文件 |
|---|---|---|
| Python 标准库：`os`、`time`、`datetime`、`argparse`、`collections` | 路径、计时、输出目录、命令行解析、分组 | 多数文件 |
| NumPy | 文本数值转换、统计、抽样、绘图数据、下采样平均 | 除 `A-KIT.py` 系列外也被工具脚本使用 |
| PyTorch | Dataset/DataLoader、神经网络、优化器、自动微分 Kalman 计算 | 除重采样脚本外的文件 |
| Matplotlib | 非交互式训练曲线和分析图 | `A-KIT_copy.py`、`A-KIT_lj.py`、`A-KIT_lj_agent.py` |

仓库中未发现 `requirements.txt`、`pyproject.toml` 或 `setup.py`，Python/PyTorch/NumPy/Matplotlib 的具体版本要求**需要进一步确认**。

## 3. 当前数据格式（对 `data_lj` 的只读核对）

`data_lj` 下存在 `data013325`、`data065517`、`data090854`、`data122041`、`data212832` 五个数据目录。每个目录均观察到以下训练相关文件：

| 文件 | 代码期望的格式 | 单样本形状/含义 |
|---|---|---|
| `imu.txt` | 首行为表头；之后为 `Time Gyro_X Gyro_Y Gyro_Z Accel_X Accel_Y Accel_Z` | 时间戳 + 6 维 IMU |
| `Vel.txt` | 首行为表头；之后为 `Time Vx Vy Vz` | 时间戳 + 3 维速度 |
| `*.IMR.ref` / `*.pos` / `*.ref` | 含 `Week GPSTime ...` 表头的参考结果 | 代码取 `GPSTime` 作键，取零基索引 9、10、11 作位置 |
| `Phi.txt` | `WeekSecond Phi`，时间戳与矩阵第一行同行，后接余下矩阵行 | `15 × 15` 状态转移矩阵 |
| `H.txt` | 同类多行格式 | `3 × 15` 观测矩阵 |
| `R.txt` | 同类多行格式 | `3 × 3` 观测噪声协方差 |
| `StateP.txt` | 同类多行格式 | `15 × 15` 状态协方差 |
| `StateX_new.txt` | 同类多行格式 | `15 × 1` 状态向量 |
| `stateX.txt` | 名称大小写及内容版本与 `StateX_new.txt` 不同 | 旧脚本使用；确切语义差异**需要进一步确认** |
| `Inno.txt` | 同类多行格式 | `3 × 1` 新息向量 |
| `Q.txt` | 同类多行格式 | `15 × 15` 原始过程噪声矩阵 |

参考文件表头表明零基索引 9～11 实际是 `X-ECEF`、`Y-ECEF`、`Z-ECEF`。`load_truePos` 中的局部变量却命名为 `x_ll/y_ll/z_ll`，相关注释也有 “X-LL” 表述；实际读取的是 ECEF 坐标，而不是表头稍后的 X/Y/Z-LL 字段。

数据只读样例：

```text
# imu.txt
Time              Gyro_X     Gyro_Y     Gyro_Z    Accel_X    Accel_Y    Accel_Z
264810.155733    0.03974   -0.02358   -0.01388   -0.17111   -4.75439    8.49449

# Vel.txt
Time Vx Vy Vz
265190.013417 1.458396 0.967825 -0.257914

# Inno.txt（3×1 多行矩阵）
WeekSecond H
265191   8.7343217e-02
         3.0187539e-01
         3.2678135e-01
```

本文档编写过程中仅列目录、读取表头和少量样例，没有修改 `data_lj` 中任何文件。

## 4. 主训练流程

### 4.1 高层调用流程

以依赖闭合的 `A-KIT_lj_agent.py` 为例：

```text
python Code/A-KIT_lj_agent.py
             |
             v
          main()
             |
             +--> 设置随机种子、超参数、设备、固定数据目录
             |
             +--> MultiDatasetIMU(data_dirs)
             |       |
             |       +--> akit_utils_agent.MultiDatasetIMU
             |               +--> 读取/按秒分组 IMU
             |               +--> 按 IMU 时间匹配 Vel
             |               +--> 读取参考位置与 7 类矩阵
             |               +--> 对所有时间键求交集
             |               +--> 生成每秒一个样本
             |
             +--> random_split(80% train, 20% val)
             +--> 统计训练集 9 通道均值/标准差
             +--> DataLoader
             +--> Network + Adam + ReduceLROnPlateau
             +--> 可选/当前默认开启的单批 Q 对角线过拟合自检
             |
             +--> epoch 循环
             |       +--> 标准化 IMU
             |       +--> Network(IMU, P) -> 15 维 raw output
             |       +--> softplus + 偏置/缩放 -> Q_diag
             |       +--> kalman_filter(..., Q_diag, Q_origin)
             |       +--> position MSE -> backward -> clip -> Adam step
             |       +--> 验证 -> scheduler -> early stopping
             |
             +--> 保存日志、统计、曲线、best/final model、训练集预测
```

### 4.2 数据读取、对齐和预处理

`akit_utils_agent.MultiDatasetIMU` 的实际流程如下：

1. 枚举位置参考文件，选取 `os.listdir()` 返回结果中第一个后缀为 `.imr.ref`、`.pos` 或 `.ref` 的文件。若同目录存在多个候选，选择哪一个没有显式排序，**需要进一步确认**。
2. `_load_and_group_imu_data()` 读取 IMU，按 `int(timestamp)` 分组，仅保留样本数严格等于 `seq_len`（默认 100）的整数秒。
3. `_load_vel_data_by_imu_timestamps()` 读取速度，在同一整数秒内为每个 IMU 时间戳寻找最近速度时间戳：
   - 时间差 `< 0.002` 秒时使用最近速度；
   - 匹配失败但本秒已有值时复制上一个速度；
   - 本秒第一个值就匹配失败时填三维零；
   - 最终只要构造出 100 个值，该秒就会保留。因此“完整 100 Hz Vel”不等价于 100 个原始速度点全部成功匹配。
4. `load_truePos()` 等待参考文件中出现同时含 `Week` 和 `GPSTime` 的表头；数据行以 `GPSTime` 为键，以 ECEF 三维位置为值。
5. `load_matrix_file()` 支持“时间戳与第一行矩阵同行”和“时间戳单独一行”两种多行格式。每行不足 `cols` 时补零，超出时截断。
6. 求以下键的交集：

```text
完整 IMU 秒 ∩ 构造完成的 Vel 秒
∩ pos ∩ Phi ∩ H ∩ R ∩ StateP ∩ StateX_new ∩ Inno ∩ Q
```

7. 每个有效整数秒生成一个样本，将同序号 IMU 六维值和速度三维值拼成 `(100, 9)`；`__getitem__()` 再转置为网络所需 `(9, 100)`。

返回字典及默认批量形状：

| 键 | 单样本形状 | DataLoader 批量形状 | dtype |
|---|---:|---:|---|
| `imu` | `(9, 100)` | `(B, 9, 100)` | `float64` |
| `time` / `matrix_time` | 标量 | `(B,)` | `float64` |
| `pos_true` | `(3,)` | `(B, 3)` | `float64` |
| `phi` | `(15, 15)` | `(B, 15, 15)` | `float64` |
| `H` | `(3, 15)` | `(B, 3, 15)` | `float64` |
| `R` | `(3, 3)` | `(B, 3, 3)` | `float64` |
| `P` / `Q` | `(15, 15)` | `(B, 15, 15)` | `float64` |
| `stateX` | `(15, 1)` | `(B, 15, 1)` | `float64` |
| `Inno` | `(3, 1)` | `(B, 3, 1)` | `float64` |

`A-KIT_lj_agent.py` 随后只用训练子集计算每个 IMU/速度通道的全局均值和标准差，并广播执行：

```python
imu_normalized = (imu - imu_mean[None, :, None]) / imu_std[None, :, None]
```

异常时静默退化为均值 0、标准差 1。主流程的训练与验证使用该归一化；小规模实验虽然计算并保存统计量，但其训练/验证循环没有实际应用该归一化，这是代码可见的不一致。

### 4.3 模型结构与张量流

五个训练脚本均包含同一族模型类，`A-KIT_lj_agent.py` 使用 `float32` 声明层，其余主要使用 `float64`；但 agent 主流程在建模后又调用 `.double()`，所以实际参数会转成 `float64`。

```text
输入 IMU/Vel
(B, 9, 100)
      |
      v
Conv1d(9 -> 64, kernel=10, stride=10)
(B, 64, 10)
      |
      v  transpose
(B, 10 patches, 64 embedding)
      |
      v
2 × SAB encoder（自注意力 + 残差 + LayerNorm + FFN）
(B, 10, 64)
      |
      v
Decoder 1: PMA(k=1) + SAB
(B, 1, 64)
      |
      v
Decoder 2: PMA(k=1) + SAB
(B, 1, 64)
      |
      v  flatten
(B, 64)
      |
      v
Linear(64,32) -> LeakyReLU -> Dropout -> Linear(32,15)
      |
      v
(B, 15) 过程噪声缩放输出
```

主要类：

| 类 | 作用 | 关键实现 |
|---|---|---|
| `MultiHeadAttention` | Q/K/V 线性投影后执行多头缩放点积注意力 | embedding 64、4 头时每头 16 维；缩放使用 `sqrt(embedding_size)`，而不是常见的 `sqrt(head_dim)` |
| `MAB` | Multihead Attention Block | `MHA(X,Y,Y)`，两次残差 + LayerNorm，中间为两层前馈网络 |
| `SAB` | Set Attention Block | 调用 `MAB(X,X)` 做自注意力 |
| `PMA` | Pooling by Multihead Attention | 学习 `k` 个 seed 向量；本项目 `k=1`，把 patch 集合池化为一个表示 |
| `Decoder` | 池化后再执行 SAB | `SAB(PMA(x))` |
| `SetTransformer` | 多层 SAB 编码器与 Decoder | 构造了 `fc_out`，但各版本 `forward()` 都没有使用它 |
| `PatchEmbedding` | 以一维卷积切分时间 patch | 默认 10 点一块且步长 10，100 点得到 10 个 patch |
| `Network` | 完整特征提取和 15 维输出头 | `forward(x, Q)` 接收 `Q`/`P` 参数，但 `clean/copy/lj/agent` 版本没有使用该参数 |

`A-KIT.py` 的 `Network` 是例外：它试图用输出对角阵与传入的 `Q[:, :, :, -2]` 批量相乘。但主流程传入的是 `P.unsqueeze(-1)`，最后一维长度为 1，索引 `-2` 会越界；详见风险章节。

### 4.4 可微 Kalman 流程

较早版本和 agent 工具版的共同数学主线为：

```text
x_pred = Phi @ stateX
P_pred = Phi @ P @ Phi^T + Q_effective
S      = H @ P_pred @ H^T + R + jitter * I
K      = P_pred @ H^T @ inv(S)
x_post = x_pred + K @ Inno
输出    = x_post 的前 3 维
```

`akit_utils_agent.kalman_filter()` 对 `Q_diag` 支持两种输入：

| `Q_diag` 形状 | `Q_effective` 构造方式 |
|---|---|
| `(B, 15)` | 取 `Q_origin` 对角线，与 15 维缩放量相乘，再用 `diag_embed` 构造纯对角矩阵；原 `Q_origin` 非对角元素不保留 |
| `(B, 15, 15)` | `Q_origin * Q_diag`，逐元素相乘 |

agent 主流程传入 `(B, 15)`，所以走第一种路径。函数内部统一转 `float64`，通过 PyTorch 运算保留自动微分图。它还计算 Joseph 形式的 `P_post` 以提高半正定稳定性，但 `P_post` 不返回，也不参与后续样本；每个样本仍是独立的单步更新，不是跨时间递推的 Kalman 序列。

损失为：

```python
loss = mean((pos_predict - pos_true) ** 2)
```

训练信号由位置误差经 Kalman 矩阵运算反传到网络预测的过程噪声缩放量。代码没有直接以 `Q_origin` 对角线作为常规训练标签；它只在单批过拟合自检中把该对角线当临时监督目标。

状态 15 个维度的完整物理定义、`Inno` 的生成方式、`Q_origin` 的单位，以及 `stateX` 是否严格代表“上一时刻状态”，无法仅从这些 Python 文件确认，**需要进一步确认**。

## 5. 各文件详细说明

### 5.1 `Code/A-KIT.py`

用途：最早期的自包含训练原型，意在用连续 100 个 IMU+速度点预测过程噪声，并用位置 MSE 训练。

| 符号 | 类型 | 作用/备注 |
|---|---|---|
| `main()` | 函数 | 使用 5 个占位数据路径，建立 DataLoader、Network、Adam，训练 50 轮 |
| `example_usage()` | 函数，定义两次 | 展示多目录数据加载与时间对齐；后一次定义覆盖前一次，但内容基本相同 |
| `check_time_alignment()` | 函数，定义两次 | 检查所需文件并尝试构建数据集；后一次定义覆盖前一次 |
| `MultiHeadAttention`、`MAB`、`SAB`、`PMA`、`Decoder`、`SetTransformer`、`PatchEmbedding`、`Network` | 模型类 | 构成卷积 Patch Embedding + Set Transformer + 输出头 |
| `load_truePos()` | 函数 | 简单文本位置读取器；包含字母的数据行会被跳过 |
| `load_matrix_file()` | 函数 | 假定一个矩阵完全位于一行；不适配当前 `data_lj` 的多行矩阵 |
| `MultiDatasetIMU` | Dataset | 以完整浮点时间交集形成长度 100 的滑动窗口 |
| `IMUDataset` | Dataset | 前半先包装 MultiDataset；随后同类中重复定义的 `__len__`/`__getitem__` 覆盖包装逻辑并访问未初始化字段 |
| `kalman_filter()` | 函数 | 单步可微 Kalman 更新，`P_pred` 直接加 `Q_diag`，参数 `Q_origin` 未使用 |

直接运行时，`if __name__ == "__main__": main()` 位于所有模型和 Dataset 类定义之前。解释器调用 `main()` 时 `MultiDatasetIMU` 尚未定义，会首先触发 `NameError`。即便调整入口顺序，仍有占位路径、样本字典缺少 `time`、`Network` 对 `Q` 的越界索引以及 `IMUDataset` 覆盖方法等问题。

### 5.2 `Code/A-KIT_clean.py`

用途：对早期原型进行去重和整理，入口移到文件末尾，数据读取改为“每整数秒必须恰好 `seq_len` 点”。

| 符号组 | 作用 |
|---|---|
| `main()` | 从 5 个相对 `data_lj/...` 路径读数据，batch=32、epoch=50、lr=1e-4；无验证集和文件输出 |
| 八个模型类 | 与上述共同模型结构一致；`Network.forward()` 忽略传入的 `Q` |
| `load_truePos()` / `load_matrix_file()` | 简化位置和单行矩阵读取 |
| `MultiDatasetIMU` / `IMUDataset` | IMU、Vel 分别按整数秒分组，只接受恰好 100 点的秒，然后对所有 1 Hz 数据求交集 |
| `kalman_filter()` | 使用 `Phi P Phi^T + Q_diag @ Q_origin` 的过程噪声项 |
| `example_usage()` / `check_time_alignment()` | 示例和文件/对齐检查，不被 `main()` 调用 |

与当前数据的主要不兼容点：读取 `StateX.txt`，而只读观察到的是 `StateX_new.txt` 和小写 `stateX.txt`；简化参考读取器会因实际数据行含 `Fixed` 等字母而跳过数据；矩阵读取器假设单行容纳全部元素，而当前矩阵是多行格式。

### 5.3 `Code/A-KIT_copy.py`

用途：在 clean 版基础上适配更复杂的参考/矩阵格式，增加训练监控、图表、文本报告和模型保存。

| 符号 | 作用 |
|---|---|
| 模块级 `os.environ.setdefault(...)` | 在导入数值库前允许重复 OpenMP，并把 OMP/MKL 线程限制为 1 |
| `main()` | 检查 4 个 Windows 绝对路径，训练 50 轮；记录 loss、耗时、Q、位置误差，周期性绘图并生成最终报告 |
| 模型类 | 与 clean 版基本一致，双精度，`Network.forward()` 忽略 `Q` |
| `load_truePos()` | 识别 `Week GPSTime` 表头并读取 ECEF 三维位置 |
| `load_matrix_file()` | 支持当前多行矩阵，并对短行补零 |
| `MultiDatasetIMU` | 读取 `StateX_new.txt`；IMU 严格按秒 100 点；Vel 对 IMU 时间做 2 ms 最近邻匹配/填充 |
| `kalman_filter()` | `Q_diag @ Q_origin` 加入先验协方差；以两个独立的 5% 随机事件打印输入或中间矩阵调试信息 |
| `generate_training_plots()` | 输出 2×3 子图：loss、耗时、位置误差、Q 演化、batch loss、学习率 |
| `generate_final_report()` | 保存 `final_model.pth`、详细分析图和 `training_report.txt` |

输出目录名为 `training_results_YYYYmmdd_HHMMSS`。脚本硬编码 Windows `F:\...\data_lj`，当前环境不能直接使用该配置。

### 5.4 `Code/A-KIT_lj.py`

用途：将数据读取/Kalman 抽到 `akit_utils`，将合并数据随机划分为 80% 训练、20% 验证，增加轻量输出开关和 `--small` 实验入口。

模块级全局变量：

| 变量 | 默认值 | 作用 |
|---|---:|---|
| `SAVE_PRED_PER_EPOCH` | `False` | 是否写验证逐样本输出相关文件 |
| `SAVE_VALID_PLOT` | `False` | 是否写验证 X 分量对比图 |
| `GENERATE_TRAIN_PROGRESS` | `False` | 是否每 10 轮调用训练过程绘图 |
| `SAVE_DETAILED_ANALYSIS` | `False` | 是否生成完整最终报告/详细分析图 |

主要函数与类：

| 符号 | 作用 |
|---|---|
| `main()` | 4 个数据集、50 轮训练；每轮验证；写 `Test.txt`；训练结束写 loss 曲线、训练集预测和模型 |
| 八个模型类 | 本文件内置双精度模型实现 |
| 六个工具同名符号 | 转发/继承 `akit_utils` 中的实现 |
| `generate_training_plots()` / `generate_final_report()` | 与 copy 版同类的绘图和报告功能 |
| `run_small_scale_experiment()` | 随机最多取 `max_samples`，固定按 80/20 划分，运行短训练/验证 |

`run_small_scale_experiment()` 参数 `val_ratio` 没有用于划分，代码固定为 80/20；导入的 `csv`、`shutil` 未使用。当前仓库没有 `Code/akit_utils.py`，所以脚本会在导入阶段失败。缺失文件是否本应由 `akit_utils_agent.py` 替代，**需要进一步确认**。

### 5.5 `Code/A-KIT_lj_agent.py`

用途：在 lj 版上集中诊断“Q 输出/梯度是否有效”，是功能最多的训练实验版本。

新增全局量除四个输出开关外还有：

| 变量 | 默认值 | 作用 |
|---|---:|---|
| `SCALE_Q` | `1.0` | 对网络产生的 Q 对角缩放量做全局乘法，注释称用于增强梯度信号 |

`main()` 的关键增强：

- 固定 Torch/NumPy 随机种子为 42；未为 DataLoader 单独提供 generator。
- 学习率改为 `5e-3`。
- 只用训练子集计算 9 通道均值/标准差，并保存 `imu_mean.npy`、`imu_std.npy`。
- `RUN_OVERFIT_MAIN=True`：正式训练前默认进行 200 步单批过拟合，自检目标是 `Q_origin` 对角线。
- 使用 Adam、梯度范数裁剪 1.0、`ReduceLROnPlateau(factor=0.5, patience=5)`。
- 验证 loss 连续 10 轮不改善时早停，并保存 `best_model.pth`。
- 大量输出 Q 分布、计算图、参数梯度、Kalman 状态更新等诊断信息。
- 写 `batch_metrics.csv`、`epoch_metrics.csv`、`Test.txt`、归一化统计、loss 图、训练集预测图和最终模型。

模型与 lj 版的主要差异：层声明为 `float32`，最终输出头没有 ReLU；主流程之后调用 `.double()`，实际又把整网转换为双精度。正式训练使用 `softplus(raw) + 1.0`，验证相同；训练结束生成 `TrainOut` 时只使用 `softplus(raw)`，少了 `+1.0`，前后不一致。

`batch_metrics.csv` 的表头声明 9 列：

```text
epoch,batch,train_loss,Q_min,Q_mean,Q_max,grad_Q_norm,param_grad_norm,pos_error_mean
```

实际每行仅写 5 个值：epoch、batch、loss、Q mean、总梯度范数。后续消费者若按表头解析会发生列语义错位。

`run_small_scale_experiment()` 支持额外的 `run_overfit`、`overfit_steps`、`overfit_lr`，但同样忽略 `val_ratio` 并固定 80/20。命令行 `--small` 当前会开启 500 步单批过拟合。

### 5.6 `Code/akit_utils_agent.py`

用途：为 agent 版提供共享工具，是唯一被另一个现存 Python 文件直接导入的仓库内模块。

| 符号 | 作用 |
|---|---|
| `load_truePos()` | 读取带 IE 风格头部的参考文件，返回 `GPSTime -> ECEF xyz` 字典 |
| `load_matrix_file()` | 读取多行矩阵，支持两种时间戳布局，短行补零 |
| `check_time_alignment()` | 检查 9 个必需文件，实例化 Dataset 并报告是否有样本 |
| `MultiDatasetIMU` | 完整的多目录数据加载、IMU 分组、Vel 匹配、时间交集与张量转换实现 |
| `IMUDataset` | 用单目录列表包装 `MultiDatasetIMU` |
| `kalman_filter()` | batch-aware、纯 Torch、双精度、可微的单步 Kalman 更新 |

其他实现细节：

- 导入的 `defaultdict` 没有使用。
- 数据集代码计算 `baseline_pos/baseline_time`，但随后明确不对 `pos_true` 或 `stateX` 做基线变换，因此该计算不影响样本。
- `kalman_filter()` 计算 `P_post`，但不返回或传递到下一时刻。
- `S` 使用 `1e-12 I` 抖动；函数开头的 `eps=1e-8` 没有实际用于该表达式。
- Kalman 每次调用都会尝试打印第一个样本前三个状态的 `Ori -> Pre + Upd = Pst`，训练时输出量很大。
- 注释称可能使用 `torch.linalg.solve`，实际实现使用 `torch.linalg.inv`，失败后回退 `torch.inverse`。

### 5.7 `Code/data_test.py`

用途：独立复刻 copy 版数据读取器，对路径、样本数、首样本和前 5 个样本的各字段形状进行检查；不训练模型。

| 符号 | 作用 |
|---|---|
| `load_truePos()` | 带更详细诊断打印的参考文件读取器 |
| `load_matrix_file()` | 多行矩阵读取器，记录并打印读取数量 |
| `MultiDatasetIMU` | IMU 严格分秒、Vel 2 ms 最近邻/填充、矩阵时间求交集 |
| `test_data_loading()` | 使用 4 个硬编码 Windows 数据目录执行完整性检查 |

文件导入了 `torch.nn`、`DataLoader`、`torch.optim`，但测试流程没有使用这些符号。它读取 `StateX.txt`；当前数据目录观察到的是 `StateX_new.txt` 和小写 `stateX.txt`，在区分大小写的文件系统上会导致对应字典为空、最终没有对齐样本。入口路径也写死为 Windows 路径。

### 5.8 `Code/resample_500_to_100.py`

用途：独立的文本降采样工具，会写入用户指定的输出文件；它不是训练入口自动调用的预处理步骤。

| 函数 | 作用 |
|---|---|
| `resample_file()` | 对单文件按整数秒分组，使用 `pick` 或 `avg` 输出目标频率 |
| `process_directory()` | 递归处理目录中匹配文件，输出同目录 `*_100hz.txt` |
| `resample_pair_imu_vel()` | 先按浮点时间戳完全相等配对 IMU/Vel，再同步降采样并写两个输出 |
| `main()` | 解析三种互斥使用方式：单文件、目录、IMU/Vel 对 |

`resample_file()` 的策略：

1. 要求 `source_rate % target_rate == 0`。
2. 若至少一个整数秒恰好有 `source_rate` 点，则只处理所有严格等于源频率的秒；其他秒不会进入弹性策略。
3. 若整个文件没有任何严格匹配秒，则对每个不少于 `target_rate` 点的秒执行等间隔 pick 或分段平均。
4. 保留输入首行表头，以指定浮点格式写新文件。

`resample_pair_imu_vel()` 先要求 IMU 和 Vel 的 Python 浮点时间戳完全相等；这比训练 Dataset 的 2 ms 最近邻条件严格。若源频率不能整除目标频率但某秒恰好 `n == source_rate`，函数仍用整数除法 factor，可能不能均匀覆盖全部样本；该函数只断言 `source_rate >= target_rate`，没有像 `resample_file()` 一样断言整除。

运行示例：

```bash
# 单文件
python3 Code/resample_500_to_100.py \
  --input /path/to/imu_500hz.txt \
  --output /path/to/imu_100hz.txt \
  --method pick

# 目录递归处理（会在原目录旁写 *_100hz.txt）
python3 Code/resample_500_to_100.py \
  --dir /path/to/dataset \
  --pattern imu.txt \
  --method avg

# IMU/Vel 同步配对处理
python3 Code/resample_500_to_100.py \
  --imu /path/to/imu.txt --vel /path/to/Vel.txt \
  --out_imu /path/to/imu_100hz.txt --out_vel /path/to/Vel_100hz.txt
```

以上命令会写数据文件。对仓库 `data_lj` 使用前应明确指定仓库外输出路径；本次任务没有运行这些写操作。

## 6. 程序入口、输入输出与运行方式

### 6.1 入口汇总

| 命令 | 实际行为 | 当前仓库状态 |
|---|---|---|
| `python3 Code/A-KIT.py` | 早期训练 | 会因入口早于类定义而失败 |
| `python3 Code/A-KIT_clean.py` | clean 版训练 | 相对路径要求从项目根运行，但读取器与当前数据格式不兼容 |
| `python3 Code/A-KIT_copy.py` | 带报告训练 | 需修改/提供代码中 Windows 绝对路径对应环境 |
| `python3 Code/A-KIT_lj.py` | train/val 训练 | 缺失 `akit_utils.py`，导入即失败 |
| `python3 Code/A-KIT_lj.py --small` | 小规模训练 | 同上 |
| `python3 Code/A-KIT_lj_agent.py` | agent 诊断增强训练 | 依赖存在，但数据路径写死为 Windows 路径 |
| `python3 Code/A-KIT_lj_agent.py --small` | agent 小规模训练 | 同上；默认额外执行 500 步 overfit |
| `python3 Code/data_test.py` | 数据读取/形状检查 | Windows 路径写死，且 `StateX.txt` 命名与当前 Linux 数据不匹配 |
| `python3 Code/resample_500_to_100.py ...` | 下采样 | 参数完整时可独立运行并写指定输出 |

训练脚本没有命令行参数用于覆盖数据根目录、batch size、epoch 或学习率。若要运行，需先确认正式版本和数据目录配置方式；本文档遵守“不修改业务代码”要求，没有进行这些调整。

### 6.2 训练输入

- 文件输入：一个或多个数据目录中的 IMU、速度、参考位置及 7 类矩阵文件。
- 网络输入：每样本 `(9, 100)` 的 IMU+Vel 张量。
- Kalman 固定输入：`Phi/H/R/P/Inno/stateX/Q_origin`。
- 监督目标：三维参考 ECEF 位置。

### 6.3 训练输出

不同版本输出不同：

| 输出 | copy | lj | lj_agent |
|---|:---:|:---:|:---:|
| 时间戳训练目录 | 是 | 是 | 是 |
| `final_model.pth` | 是 | 是 | 是 |
| `best_model.pth` | 否 | 否 | 是 |
| `Test.txt` | 否 | 是 | 是 |
| `Loss_vs_Epoch.png` | 间接在综合图 | 是 | 是 |
| `TrainOut.txt` / 训练 X 对比图 | 否 | 是 | 是 |
| `imu_mean.npy` / `imu_std.npy` | 否 | 否 | 是 |
| batch/epoch CSV | 否 | 否 | 是 |
| 详细图和文本报告 | 默认是 | 由开关控制 | 由开关控制 |

模型文件只保存 `state_dict` 和少量训练元数据，没有统一保存完整模型构造参数、数据划分索引、优化器状态、scheduler 状态或所有随机状态。如何加载并执行独立推理没有对应入口，**需要进一步确认**。

## 7. 已确认的可运行性与一致性风险

以下均来自当前代码静态分析或与只读数据格式的直接对照，不是对未知业务规则的猜测：

| 优先级 | 位置 | 已确认问题/风险 |
|---|---|---|
| 高 | `A-KIT.py` | `main()` 在类定义前执行，直接运行首先出现 `NameError` |
| 高 | `A-KIT.py` | `MultiDatasetIMU` 样本未写 `time`，但 `__getitem__` 读取它；`IMUDataset` 方法被重复定义覆盖 |
| 高 | `A-KIT.py` | `Network` 对最后一维长度为 1 的输入索引 `-2`，会越界 |
| 高 | `A-KIT_lj.py` | 导入的 `akit_utils.py` 不存在 |
| 高 | 多个训练/测试脚本 | 数据根目录硬编码为另一台机器的 Windows 绝对路径；当前环境不可直接运行 |
| 高 | `A-KIT_clean.py` | 位置/矩阵解析假设与当前 IE 参考文件、多行矩阵格式不匹配 |
| 高 | `data_test.py` | 请求 `StateX.txt`，当前区分大小写环境中没有该精确文件名 |
| 中 | agent 主/小流程 | `val_ratio` 参数未使用，始终固定 80/20 |
| 中 | agent 主流程 | TrainOut 阶段 Q 变换少 `+1.0`，与训练/验证不一致 |
| 中 | agent 主流程 | batch CSV 表头 9 列，实际数据 5 列 |
| 中 | agent 小流程 | 计算归一化统计但不对训练/验证输入应用 |
| 中 | 工具 Dataset | Vel 匹配失败会前向填充/补零，仍可能被计为该秒完整 |
| 中 | 模型类 | `SetTransformer.fc_out` 未使用，`Network.forward` 的第二参数多数版本未使用 |
| 中 | Kalman 工具 | 计算 `P_post` 但不跨样本递推；实际是独立单步更新 |
| 低 | 注意力 | 缩放分母使用 `sqrt(embedding_size)` 而不是常见实现的 `sqrt(head_dim)`；是否有意如此**需要进一步确认** |
| 低 | 多文件 | 多处捕获宽泛 `Exception` 后静默 `pass`，可能掩盖归一化、绘图或输出错误 |

## 8. 需要进一步确认的事项

1. 正式维护和运行的训练入口究竟是 `A-KIT_copy.py`、`A-KIT_lj.py` 还是 `A-KIT_lj_agent.py`。
2. `A-KIT_lj.py` 所需的 `akit_utils.py` 是否遗失，还是应改用 `akit_utils_agent.py`。
3. 15 维状态向量每一维的物理含义、坐标系、单位及误差状态/绝对状态定义。
4. `StateX_new.txt` 与 `stateX.txt` 的来源、时间偏移和业务差异。
5. 位置监督是否确实应使用 ECEF 字段；当前代码变量/注释中的 “LL” 与实际字段不一致。
6. IMU 和 Vel 未在 2 ms 内匹配时，前向填充或补零是否满足业务要求。
7. 数据划分是否允许随机打散同一轨迹的相邻秒。当前随机样本级 80/20 划分可能让同一数据目录的相邻时刻同时进入训练和验证；是否需要按轨迹/目录划分无法从代码确认。
8. Q 网络输出究竟表示过程噪声本身、对 `Q_origin` 的对角缩放，还是完整矩阵的比例因子。不同历史版本的公式不完全一致。
9. 训练位置误差输出标注为“米”，这依赖 ECEF 坐标确以米为单位；参考文件表头和常见格式支持这一理解，但 Python 代码本身未显式声明单位。
10. 生产推理、模型加载、复现实验所需的固定依赖版本和配置管理方式。

## 9. 总结

代码库体现了从自包含原型到诊断增强训练实验的迭代过程。稳定的概念主线是“100 Hz IMU+速度窗口 → Patch/Set Transformer → 15 维 Q 缩放 → 单步可微 Kalman → 三维位置 MSE”。当前最完整且仓库内依赖闭合的实现是 `A-KIT_lj_agent.py` 与 `akit_utils_agent.py`，但仍需要解决硬编码路径、版本选择和若干训练/输出一致性问题后，才能形成可复现的标准运行入口。
