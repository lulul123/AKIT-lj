# GNSS 间隔等效 Phi 与 Q 累计方案

## 1. 目的

本方案用于解决当前 A-KIT 深度学习流程中的一个关键问题：原 C++ 松组合滤波器以约 100 Hz 执行协方差预测，而 GNSS 通常以约 1 Hz 更新。如果只在 GNSS 到达前最后一个 0.01 s 预测步替换过程噪声 Q，网络只能直接改变最后一步注入的过程噪声，对最终先验协方差和卡尔曼增益的影响可能很小。

本方案将一个 GNSS 更新间隔内的全部高频 Phi 和 Q 精确合并为一组等效矩阵：

- `P_start`：区间开始时的验后协方差；
- `Phi_acc`：整个 GNSS 间隔的累计状态转移矩阵；
- `Q_acc`：整个 GNSS 间隔的累计等效过程噪声；
- `P_pred`：当前 GNSS 更新前的先验协方差。

四者始终满足：

$$
P_{pred}=\Phi_{acc}P_{start}\Phi_{acc}^{T}+Q_{acc}
$$

深度学习模型使用一个 GNSS 间隔内的 IMU 和速度序列调整 `Q_acc`，而不是只调整最后 0.01 s 的 Q。

## 2. 变量命名与固定含义

为了避免预测前、预测后以及单步量和累计量混淆，本文统一使用以下名称。

| 名称 | 维度 | 固定含义 |
|---|---:|---|
| `P_start` | 15×15 | 上一次成功量测更新后的验后协方差 |
| `Phi_i` | 15×15 | 第 i 个 IMU 传播步的单步状态转移矩阵 |
| `Q_i` | 15×15 | 第 i 个 IMU 传播步的单步离散过程噪声，即 `m_KFTmp_Q` |
| `Phi_acc` | 15×15 | 从区间起点到当前时刻的累计状态转移矩阵 |
| `Q_acc` | 15×15 | 从区间起点到当前时刻的累计等效过程噪声 |
| `P_base` | 15×15 | 不包含过程噪声的协方差传播结果 |
| `P_pred` | 15×15 | 当前 GNSS 更新前的先验协方差 |
| `P_post` | 15×15 | 当前 GNSS 更新后的验后协方差 |

其中：

$$
P_{base}=\Phi_{acc}P_{start}\Phi_{acc}^{T}
$$

$$
P_{pred}=P_{base}+Q_{acc}
$$

## 3. 原始高频协方差预测

假设 GNSS 更新时刻为 1000.00 s 和 1001.00 s，IMU 频率为 100 Hz，则中间约有 100 个传播步：

```text
1000.00 -> 1000.01
1000.01 -> 1000.02
...
1000.99 -> 1001.00
```

原 C++ 在每个传播步执行：

$$
P_i=\Phi_iP_{i-1}\Phi_i^{T}+Q_i
$$

第一步：

$$
P_1=\Phi_1P_{start}\Phi_1^{T}+Q_1
$$

第二步：

$$
P_2=\Phi_2P_1\Phi_2^{T}+Q_2
$$

一直到第 100 步：

$$
P_{100}=\Phi_{100}P_{99}\Phi_{100}^{T}+Q_{100}
$$

`P_100` 就是 1001.00 s 量测更新前的 `P_pred`。

单步 `Q_i` 由连续时间噪声谱密度 `Qc_i` 离散化得到：

$$
Q_i=\frac{1}{2}\left(\Phi_iQ_{c,i}\Phi_i^T+Q_{c,i}\right)\Delta t_i
$$

`Qc_i` 不是上一预测步的离散 Q。其基础数值来自配置中的 `PNSD_*²`，并在每个 IMU 步根据当前姿态、导航坐标系以及是否为插值 IMU 等条件更新。历史 Q 的影响通过 P 逐步传递，而不是通过 `Qc` 或 `m_KFTmp_Q` 累计。

## 4. 为什么可以合并为一次等效预测

以两个传播步为例：

$$
P_1=\Phi_1P_{start}\Phi_1^T+Q_1
$$

$$
P_2=\Phi_2P_1\Phi_2^T+Q_2
$$

代入并展开：

$$
P_2=\Phi_2\Phi_1P_{start}\Phi_1^T\Phi_2^T+\Phi_2Q_1\Phi_2^T+Q_2
$$

定义：

$$
\Phi_{acc}=\Phi_2\Phi_1
$$

$$
Q_{acc}=\Phi_2Q_1\Phi_2^T+Q_2
$$

便得到：

$$
P_2=\Phi_{acc}P_{start}\Phi_{acc}^T+Q_{acc}
$$

该合并是矩阵公式的精确展开，不是近似。100 个传播步同样可以合并。

## 5. 累计 Phi 的算法

区间开始时初始化：

$$
\Phi_{acc,0}=I
$$

每得到一个当前单步矩阵 `Phi_i`，执行：

$$
\Phi_{acc,i}=\Phi_i\Phi_{acc,i-1}
$$

因此：

$$
\Phi_{acc,100}=\Phi_{100}\Phi_{99}\cdots\Phi_2\Phi_1
$$

C++ 伪代码：

```cpp
PhiAccNew.multiply(PhiCurrent, PhiAcc);
PhiAcc = PhiAccNew;
```

乘法顺序必须是：

```text
当前单步 Phi × 之前累计 Phi
```

不能交换顺序，因为矩阵乘法通常不可交换。

## 6. 累计 Q 的算法

区间开始时初始化：

$$
Q_{acc,0}=0
$$

每得到当前单步 `Phi_i` 和 `Q_i`，执行：

$$
Q_{acc,i}=\Phi_iQ_{acc,i-1}\Phi_i^T+Q_i
$$

第一步后：

$$
Q_{acc,1}=Q_1
$$

第二步后：

$$
Q_{acc,2}=\Phi_2Q_1\Phi_2^T+Q_2
$$

第三步后：

$$
Q_{acc,3}=\Phi_3\left(\Phi_2Q_1\Phi_2^T+Q_2\right)\Phi_3^T+Q_3
$$

C++ 伪代码：

```cpp
QAccNew = PhiCurrent * QAcc * PhiCurrent.transpose();
QAccNew += QCurrent;
QAcc = QAccNew;
```

这里 `QCurrent` 就是本次 `KFPrediction()` 已经计算完成的 `m_KFTmp_Q`。

## 7. 一个完整 GNSS 更新间隔的流程

### 7.1 上一次量测更新成功后

假设 1000.00 s 的 GNSS 更新成功，C++ 得到验后协方差：

$$
P_{post,1000}
$$

开始新区间时执行：

```cpp
PStart = P_post_1000;
PhiAcc = I;
QAcc = 0;
```

即：

$$
P_{start}=P_{post,1000}
$$

### 7.2 每个 IMU 传播步

原 C++ 滤波仍然正常执行单步预测：

$$
P_i=\Phi_iP_{i-1}\Phi_i^T+Q_i
$$

同时额外更新两个累计器：

$$
\Phi_{acc}\leftarrow\Phi_i\Phi_{acc}
$$

$$
Q_{acc}\leftarrow\Phi_iQ_{acc}\Phi_i^T+Q_i
$$

累计器只是记录等效量，不替代原有 C++ 滤波。

### 7.3 到达下一次 GNSS 时刻

假设 1001.00 s 有有效 GNSS 量测，在 GLS 量测更新前，当前累计量应满足：

$$
P_{pred,cpp}\approx\Phi_{acc}P_{start}\Phi_{acc}^T+Q_{acc}
$$

此时输出一组数据：

```text
time
P_start
Phi_acc
Q_acc
P_pred_cpp
H
R
Inno
```

其中 `P_pred_cpp` 是 C++ 逐步预测得到的先验协方差，主要用于验证累计公式是否正确。

### 7.4 当前 GNSS 更新完成后

如果 1001.00 s 量测更新成功，得到新的：

$$
P_{post,1001}
$$

下一段 1001.00 s 到后续 GNSS 时刻使用：

```cpp
PStart = P_post_1001;
PhiAcc = I;
QAcc = 0;
```

## 8. GNSS 缺失或更新被拒绝

累计器不应在每个整数秒机械重置，而应按照“成功量测更新区间”管理。

如果 1000.00 s 更新成功，但 1001.00 s 没有 GNSS，下一次有效量测在 1003.00 s，则：

```text
区间起点：1000.00
区间终点：1003.00
累计时长：约 3 s
传播步数：约 300
```

这时 `Phi_acc` 和 `Q_acc` 应连续累计到 1003.00 s，不能在 1001.00 s 或 1002.00 s 清零。

如果 1001.00 s 虽然有 GNSS 数据，但量测被质量控制拒绝，也没有产生可信的新验后协方差。因此通常不应把该时刻作为下一个区间的 `P_start`。是否输出被拒绝样本，应根据训练目标单独设计。

## 9. 建议的输出文件

建议避免继续使用含义模糊的 `StateP.txt`、`Phi.txt` 和 `Q.txt`，改用：

```text
PStart.txt
PhiAcc.txt
QAcc.txt
PPredCpp.txt
H.txt
R.txt
Inno.txt
```

以时间戳 1001 为例：

| 文件 | 时间戳 1001 的含义 |
|---|---|
| `PStart.txt` | 上一次成功更新后、当前区间起点的验后 P |
| `PhiAcc.txt` | 从区间起点传播到 1001 的累计 Phi |
| `QAcc.txt` | 从区间起点传播到 1001 的累计等效 Q |
| `PPredCpp.txt` | C++ 原滤波传播到 1001 得到的先验 P |
| `H/R/Inno.txt` | 1001 时刻量测更新使用的参数 |

最好额外输出：

```text
start_time
end_time
step_count
total_dt
update_accepted
```

这样可以明确一个样本实际覆盖的时间范围、IMU步数以及量测是否成功接受。

## 10. 深度学习数据组织

每个训练样本建议包含：

```text
IMU + Vel 序列
P_start
Phi_acc
Q_acc_origin
H
R
Inno
state_pred
pos_true
```

当 GNSS 严格为 1 Hz、IMU 为 100 Hz 时，IMU 与速度输入形状通常为：

```text
[100, 9]
```

9 个通道为：

```text
Gyro_X, Gyro_Y, Gyro_Z,
Accel_X, Accel_Y, Accel_Z,
Vel_X, Vel_Y, Vel_Z
```

如果 GNSS 间隔不固定，应同时记录真实序列长度并使用 padding、mask 或变长序列模型，不能默认所有样本都恰好有 100 点。

## 11. 网络调整整段累计 Q

当前模型可以继续输出 15 个正缩放系数：

$$
s=[s_1,s_2,\ldots,s_{15}]
$$

最简单的方案是缩放 `Q_acc_origin` 的对角线：

$$
Q_{acc,pred}=\operatorname{diag}\left(\operatorname{diag}(Q_{acc,origin})\odot s\right)
$$

PyTorch 示例：

```python
q_origin_diag = torch.diagonal(
    Q_acc_origin,
    dim1=-2,
    dim2=-1,
)

q_pred_diag = q_origin_diag * q_scale
Q_acc_pred = torch.diag_embed(q_pred_diag)
```

这里网络调整的是整个 GNSS 间隔的累计过程噪声，而不是最后一个 0.01 s 的单步 Q。

只保留对角线会丢弃 `Q_acc_origin` 的非对角相关项。如果后续实验表明非对角项重要，可以考虑保留原非对角项，或使用保证半正定的 Cholesky 参数化预测完整矩阵。

## 12. Python 中的协方差预测

首先计算不包含过程噪声的传播结果：

```python
P_base = (
    Phi_acc
    @ P_start
    @ Phi_acc.transpose(-2, -1)
)
```

即：

$$
P_{base}=\Phi_{acc}P_{start}\Phi_{acc}^T
$$

然后加入网络调整后的整段累计 Q：

```python
P_pred = P_base + Q_acc_pred
```

即：

$$
P_{pred}=P_{base}+Q_{acc,pred}
$$

不能把 `P_base` 再代入原单步公式执行一次 `Phi_acc @ P_base @ Phi_acc.T`，否则会发生重复传播。

## 13. GNSS 量测更新

新息协方差：

```python
S = (
    H
    @ P_pred
    @ H.transpose(-2, -1)
    + R
)
```

$$
S=HP_{pred}H^T+R
$$

卡尔曼增益：

```python
K = (
    P_pred
    @ H.transpose(-2, -1)
    @ torch.linalg.inv(S)
)
```

$$
K=P_{pred}H^TS^{-1}
$$

状态改正量：

```python
state_update = K @ Inno
```

$$
\delta X=K\,Inno
$$

状态更新：

```python
state_post = state_pred + state_update
```

$$
X_{post}=X_{pred}+K\,Inno
$$

Joseph 形式协方差更新：

```python
I_KH = I - K @ H

P_post = (
    I_KH @ P_pred @ I_KH.transpose(-2, -1)
    + K @ R @ K.transpose(-2, -1)
)
```

$$
P_{post}=(I-KH)P_{pred}(I-KH)^T+KRK^T
$$

## 14. 与当前 Python 代码的主要区别

当前代码使用单步数据：

```python
P_pred = phi @ P @ phi.T + Q_effective
```

整段累计方案改为：

```python
P_pred = Phi_acc @ P_start @ Phi_acc.T + Q_acc_pred
```

形式相似，但变量含义完全不同：

| 当前实现 | 累计方案 |
|---|---|
| `phi` 是约 0.01 s 单步 Phi | `Phi_acc` 是整个 GNSS 间隔累计 Phi |
| `Q_origin` 是约 0.01 s 单步 Q | `Q_acc_origin` 是整个 GNSS 间隔累计 Q |
| `P` 是当前单步预测输入 | `P_start` 是区间起点的验后 P |

因此不能只替换文件内容而不检查变量语义。

## 15. C++ 累计器实现框架

可以在滤波类或松组合类中增加：

```cpp
Mat m_PStart;
Mat m_PhiAcc;
Mat m_QAcc;
bool m_AccInitialized;
int m_AccStepCount;
double m_AccStartTime;
```

区间开始时：

```cpp
m_PStart = m_GLSInfo.StateP;
m_PhiAcc.eye();
m_QAcc.zero();
m_AccStepCount = 0;
m_AccStartTime = currentTime;
m_AccInitialized = true;
```

每个 `KFPrediction()` 在当前 `Phi` 和 `m_KFTmp_Q` 计算完成后：

```cpp
// Q_acc,new = Phi_current * Q_acc,old * Phi_current^T + Q_current
Mat QAccNew(m_nState, m_nState);
MatrixMultiply_HPHT(
    Phi.row,
    Phi.col,
    Phi.x,
    m_QAcc.x,
    false,
    QAccNew.x
);
QAccNew.add2(m_KFTmp_Q);
m_QAcc = QAccNew;

// Phi_acc,new = Phi_current * Phi_acc,old
Mat PhiAccNew(m_nState, m_nState);
PhiAccNew.multiply(Phi, m_PhiAcc);
m_PhiAcc = PhiAccNew;

m_AccStepCount += 1;
```

在当前有效量测更新前输出累计结果；成功更新后，使用新的验后 `StateP` 初始化下一个区间。

具体落地时要根据 `Mat` 类的构造、赋值和内存管理规则调整代码，避免临时矩阵浅拷贝或维度未初始化。

## 16. 上线回 C++ 时如何使用网络 Q

训练阶段使用：

$$
P_{pred,NN}=\Phi_{acc}P_{start}\Phi_{acc}^T+Q_{acc,NN}
$$

模型部署回 C++ 后，在量测更新之前，用网络生成的 `Q_acc_NN` 构造新的先验协方差，再交给 GLS：

```cpp
PBase = PhiAcc * PStart * PhiAcc.transpose();
PAdjusted = PBase + QAccNN;
m_GLSInfo.SINSPPre = PAdjusted;
m_GLSInfo.StateP = PAdjusted;
```

随后原有 H、R、Inno、卡尔曼增益和 Joseph 更新流程保持不变。

必须确保网络输出对应当前区间，且只在量测更新前替换一次先验协方差，不能在相同区间重复加入 `Q_acc_NN`。

## 17. 必须先做的一致性验证

在引入神经网络前，先验证 C++ 累计结果是否正确。

从文件读取：

```text
P_start
Phi_acc
Q_acc_origin
P_pred_cpp
```

Python 重建：

```python
P_pred_rebuilt = (
    Phi_acc
    @ P_start
    @ Phi_acc.transpose(-2, -1)
    + Q_acc_origin
)
```

计算误差：

```python
error_max = torch.max(torch.abs(P_pred_rebuilt - P_pred_cpp))
error_rel = torch.linalg.norm(P_pred_rebuilt - P_pred_cpp) / (
    torch.linalg.norm(P_pred_cpp) + 1e-15
)
```

只有满足：

```text
P_pred_rebuilt ≈ P_pred_cpp
```

才能确认：

- `P_start` 输出时机正确；
- `Phi_acc` 乘法顺序正确；
- `Q_acc` 递推正确；
- 没有漏掉 GNSS 插值产生的传播步；
- 累计器重置时机正确。

常见错误包括：

1. 把 `Phi_acc = Phi_acc * Phi_current` 写反；
2. 在当前 GNSS 预测完成之前提前清零累计器；
3. 输出了更新后的 P，却把它当作更新前 `P_pred`；
4. 按整数秒重置，而不是按成功量测更新重置；
5. GNSS 插值时漏掉被拆分的一个传播步；
6. 使用 `P_base` 后又额外执行一次 Phi 传播。

## 18. 推荐实施顺序

建议分阶段实施，避免同时修改数据和训练代码后难以定位问题。

### 第一阶段：只实现 C++ 累计与输出

保持原滤波结果不变，只新增：

```text
PStart.txt
PhiAcc.txt
QAcc.txt
PPredCpp.txt
```

### 第二阶段：验证累计公式

确认：

$$
P_{pred,cpp}\approx\Phi_{acc}P_{start}\Phi_{acc}^T+Q_{acc}
$$

### 第三阶段：修改数据读取

让 `akit_utils_agent.py` 读取新的累计矩阵，并明确字段名：

```python
sample["P_start"]
sample["Phi_acc"]
sample["Q_acc"]
```

### 第四阶段：修改 Python 卡尔曼预测

使用：

```python
P_pred = Phi_acc @ P_start @ Phi_acc.T + Q_acc_pred
```

### 第五阶段：训练并检查敏感度

检查：

- `Q_acc_pred` 是否显著影响 `P_pred`；
- `P_pred` 是否显著影响 K；
- 损失对网络输出的梯度是否非零且量级合理；
- Q 缩放是否出现极端值；
- P 是否保持对称和半正定。

### 第六阶段：部署回 C++

在 GNSS 量测更新前，用网络累计 Q 替换原累计 Q 对先验协方差的贡献。

## 19. 最终算法摘要

每次成功量测更新后：

```text
P_start = 当前验后P
Phi_acc = I
Q_acc   = 0
```

每个 IMU 传播步：

```text
计算 Phi_i
计算 Q_i

Q_acc   = Phi_i * Q_acc * Phi_i^T + Q_i
Phi_acc = Phi_i * Phi_acc
```

下一次有效 GNSS 更新前：

```text
输出 P_start、Phi_acc、Q_acc、H、R、Inno
```

深度学习模型：

```text
一个GNSS间隔内的IMU+Vel
            ↓
网络预测Q_acc缩放量
            ↓
Q_acc_pred
```

Python 卡尔曼更新：

```text
P_base = Phi_acc * P_start * Phi_acc^T
P_pred = P_base + Q_acc_pred
S      = H * P_pred * H^T + R
K      = P_pred * H^T * inv(S)
X_post = X_pred + K * Inno
```

该方案在不向 Python 输出 100 组高频矩阵的情况下，保留了整个 GNSS 更新间隔内全部单步 Phi 和 Q 对最终先验协方差的等效影响。
