# A-KIT 训练原理

> **阅读提示（2026-10-26）**  
> 本文第 1–7 节的论文原理总结仍有效。第 8 节以后对本项目
> `PhiAcc/QAcc` 路线的描述是历史记录，已被“缩放连续时间 `Qc`
> 并逐步重放 `Phi/dt`”路线取代。当前项目逻辑以
> 当前项目的数据契约、时间戳边界和逐 Phi 重放方法以
> `数据总说明.md` 为准。

## 1. 参考文献与方法目标

本文根据 [Cohen 和 Klein (2025), Adaptive Kalman-Informed Transformer](1_Cohen%20和%20Klein%20-%202025%20-%20Adaptive%20Kalman-Informed%20Transformer.pdf) 总结 A-KIT 的训练思路，并说明它与当前项目实现的对应关系。

A-KIT 的目标是在线自适应调整 EKF 过程噪声协方差 Q，不是用神经网络取代 EKF。其核心是：

> 网络输出 Q 的缩放系数，可微 Kalman 预测和更新将状态真值损失反向传播到网络。

## 2. 论文中的基础 ES-EKF

论文定义误差状态为：

$$
\mathbf{x}_t=\mathbf{x}_e-\delta\mathbf{x}.
$$

协方差预测为：

$$
\mathbf{P}_k^-
=
\mathbf{\Phi}_{k-1}\mathbf{P}_{k-1}^+
\mathbf{\Phi}_{k-1}^T
+\mathbf{Q}_{k-1}.
$$

Kalman 增益为：

$$
\mathbf{K}_k
=
\mathbf{P}_k^-\mathbf{H}_k^T
\left(
\mathbf{H}_k\mathbf{P}_k^-\mathbf{H}_k^T+\mathbf{R}_k
\right)^{-1}.
$$

误差状态更新为：

$$
\delta\mathbf{x}_k^+
=
\mathbf{K}_k\delta\mathbf{z}_k.
$$

论文使用的简化验后协方差公式为：

$$
\mathbf{P}_k^+
=
(\mathbf{I}-\mathbf{K}_k\mathbf{H}_k)\mathbf{P}_k^-.
$$

本项目使用 Joseph 形式：

$$
\mathbf{P}_k^+
=
(\mathbf{I}-\mathbf{K}_k\mathbf{H}_k)
\mathbf{P}_k^-
(\mathbf{I}-\mathbf{K}_k\mathbf{H}_k)^T
+\mathbf{K}_k\mathbf{R}_k\mathbf{K}_k^T,
$$

数值稳定性更好。

## 3. A-KIT 网络预测什么

论文先使用基于新息窗口的自适应方法估计中间过程噪声：

$$
\widehat{\mathbf{C}}_{v,k}
=
\frac{1}{N}
\sum_{j=k-N+1}^{k}
\delta\mathbf{z}_j\delta\mathbf{z}_j^T,
$$

$$
\widehat{\mathbf{Q}}_k
=
\mathbf{K}_k
\widehat{\mathbf{C}}_{v,k}
\mathbf{K}_k^T.
$$

A-KIT 不直接输出状态，而是输出一个对角缩放矩阵：

$$
\mathbf{Q}^{A\text{-}KIT}_k
=
\operatorname{diag}(s_{k,1},\ldots,s_{k,n}).
$$

最终过程噪声为论文公式(29)：

$$
\mathbf{Q}_k
=
\widehat{\mathbf{Q}}_k
\cdot
\mathbf{Q}^{A\text{-}KIT}_k.
$$

论文用点乘符号表示缩放。在其对角输出设定下，核心意图是使用网络对中间 Q 的各状态分量做自适应缩放。

## 4. 网络输入与结构

### 4.1 论文案例的输入

INS/DVL 案例使用一秒窗口，INS频率为100 Hz，因此每个样本包含100个时刻的：

- IMU 比力；
- IMU 角速度；
- 自适应 EKF 估计的速度状态。

论文 Algorithm 2 还将中间新息协方差估计 $\widehat{\mathbf{Q}}_k$ 列入 A-KIT 调用信息。

### 4.2 Set Transformer

结构为：

```text
100点IMU+估计速度
        ↓
1D卷积Patch Embedding
        ↓
Set Attention Blocks (SAB)
        ↓
Pooling by Multihead Attention (PMA)
        ↓
前馈网络
        ↓
Q的对角缩放系数
```

多头注意力为：

$$
\operatorname{Attention}(\mathbf{Q},\mathbf{K},\mathbf{V})
=
\operatorname{Softmax}
\left(
\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_q}}
\right)\mathbf{V}.
$$

论文的输出经 ReLU 保证非负，ReLU 产生的0被替换为1。

## 5. Kalman-informed loss

### 5.1 训练时保存的 EKF 信息

从上一次有效量测更新到当前更新，论文保存：

- 上一次更新后的验后协方差 $\mathbf{P}_{k-1}^{+}$；
- 两次更新之间的所有状态转移矩阵 $\mathbf{\Phi}_k$；
- 当前量测矩阵 $\mathbf{H}_T$；
- 当前量测噪声 $\mathbf{R}_T$；
- 当前新息 $\delta\mathbf{z}_T$；
- EKF 当前名义状态 $\mathbf{x}_{EKF}^{n}$；
- 可观状态的真值 $\mathbf{x}_{GT}^{n}$。

### 5.2 用网络 Q 重放预测

论文 Algorithm 1 在一个观测间隔内重新执行协方差预测：

$$
\mathbf{P}_k^-
=
\mathbf{\Phi}_k\mathbf{P}_{k-1}^+
\mathbf{\Phi}_k^T
+\mathbf{Q}^{A\text{-}KIT}.
$$

论文算法用 `k=1,...,T` 循环重放两次有效更新之间的预测步，并在最终观测时刻 $T$ 计算：

$$
\mathbf{K}_T
=
\mathbf{P}_T^-\mathbf{H}_T^T
\left(
\mathbf{H}_T\mathbf{P}_T^-\mathbf{H}_T^T+\mathbf{R}_T
\right)^{-1},
$$

$$
\delta\mathbf{x}_T^+
=
\mathbf{K}_T\delta\mathbf{z}_T.
$$

### 5.3 恢复可观名义状态

由于论文定义 $\mathbf{x}_t=\mathbf{x}_e-\delta\mathbf{x}$，所以采用减号反馈：

$$
\mathbf{x}_{A\text{-}KIT}^{n}
=
\mathbf{x}_{EKF}^{n}
-\delta\mathbf{x}_T^+.
$$

注意：加号还是减号取决于误差状态和新息的定义。本项目 C++ 内部注释为 `X = Xp + K * Inno`，不能不核对符号约定就直接照搬论文的减号。

### 5.4 损失

论文只用外部传感器能直接观测的状态计算 MSE。INS/DVL 案例中的监督对象是速度：

$$
\mathcal{L}
=
\operatorname{MSE}
\left(
\mathbf{x}_{A\text{-}KIT}^{n},
\mathbf{x}_{GT}^{n}
\right).
$$

梯度通过整个 Kalman 链路传播：

$$
\mathcal{L}
\rightarrow
\mathbf{x}_{A\text{-}KIT}^{n}
\rightarrow
\delta\mathbf{x}_T^+
\rightarrow
\mathbf{K}_T
\rightarrow
\mathbf{P}_T^-
\rightarrow
\mathbf{Q}^{A\text{-}KIT}
\rightarrow
\theta.
$$

这就是“Kalman-informed”的核心：没有 Q 真值标签，网络由最终可观状态误差间接监督。

## 6. 论文的训练配置

INS/DVL 案例使用：

| 项目 | 论文设置 |
|---|---:|
| IMU频率 | 100 Hz |
| DVL频率 | 1 Hz |
| 输入窗口 | 1 s / 100点 |
| Patch卷积核 | 5 |
| Stride | 1 |
| Patch size | 1 |
| 卷积滤波器/潜维 | 32 |
| 注意力头 | 2 |
| FFN扩展维度 | 64 |
| SAB堆叠数 | 2 |
| PMA向量数 | 1 |
| Dropout | 0.1 |
| 优化器 | RMSprop |
| 学习率 | $10^{-5}$ |
| Weight decay | $10^{-4}$ |
| Batch size | 400 |
| Epochs | 100 |
| 新息窗口 | 5次更新（对比自适应方法，Algorithm 2也使用此类窗口） |

训练集由11段各400 s的不同动态轨迹构成，测试使用额外两段400 s轨迹，与训练段分离。

## 7. 训练阶段与部署阶段的区别

### 7.1 训练阶段

论文使用保存的 EKF 中间量重建一个观测间隔的协方差预测和最终更新，用真值状态 MSE 优化网络 Q。

从 Algorithm 1 的输入看，每个训练间隔使用保存的 $\mathbf{P}_{k-1}^{+}$、$\mathbf{\Phi}$、新息和 EKF 状态。论文没有在 Kalman-informed loss 的伪代码中说明将当前网络验后 P 跨 mini-batch 传给下一个训练样本。因此不应仅根据论文将训练解读为完整的跨样本 BPTT 闭环。

### 7.2 部署阶段

论文 Algorithm 2 明确将训练后 A-KIT 接入连续 EKF：

1. 按 IMU 频率执行 INS 机械编排和 EKF 预测；
2. 有效 DVL 到达时执行量测更新和 INS 改正；
3. 新息窗口充足后估计 $\widehat{\mathbf{Q}}_k$；
4. A-KIT 输出缩放矩阵；
5. 得到新 Q，并影响后续连续预测与更新。

所以论文的最终性能来自网络嵌入真实连续 EKF 后的闭环轨迹，不是只报告离线单步 loss。

## 8. 论文方法与当前项目的对应

| 论文 A-KIT | 当前 GNSS 项目 |
|---|---|
| INS/DVL，DVL速度1 Hz更新 | INS/GNSS，GNSS位置更新 |
| 12维误差状态 | 15维误差状态 |
| 监督可观速度 | 监督ECEF位置 |
| 保存上次验后P和区间内所有Phi | C++预先累计为 `PStart/PhiAcc/QAcc` |
| 训练loss中按预测步重放P | Python用区间累计 `PhiAcc/QAcc`只预测一次 |
| 网络缩放新息法估计的 $\widehat Q$ | 网络缩放C++模型累计 `QAcc_origin` 的对角线 |
| 输入1 s IMU+估计速度 | 输入1 s IMU+速度 |
| ReLU正缩放，0替换为1 | `softplus`；正式路径额外 `+1` |
| 通过可观状态MSE间接学Q | 通过ECEF位置MSE间接学Q |
| 部署时接入连续EKF闭环 | 当前Python训练尚未将网络验后状态/P传到下一样本 |

当前项目将高频递推等效压缩为：

$$
\mathbf{P}_{pred}
=
\mathbf{\Phi}_{acc}\mathbf{P}_{start}
\mathbf{\Phi}_{acc}^T
+\mathbf{Q}_{acc,pred}.
$$

当 `QAcc_pred` 确实是网络过程噪声经过各高频状态转移后的等效累计量时，上式与论文 Algorithm 1 在一个观测间隔内循环传播 P 数学等价。当前代码是直接缩放累计后的 `QAcc_origin` 对角线；这仍是有效的区间级 Q 参数化，但通常不严格等同于在每个 100 Hz 预测步缩放 Q 后再累计。

## 9. 当前项目不能直接照搬的部分

1. **状态反馈符号**：论文用 $x_{true}=x_{est}-\delta x$，当前 C++ 代码使用 `Xp + K*Inno`，必须以本项目实际定义为准。
2. **Q基准不同**：论文缩放新息方法估计的 Q，当前项目缩放 C++ 物理模型累计 Q。
3. **可观状态不同**：论文监督 DVL 可观速度，当前项目监督 GNSS 位置。
4. **迭代 GLS**：当前 C++ 可执行多次 GLS 线性化，Python只执行一次更新，必须固定使用同一次迭代的验前状态、H、R和新息。
5. **数据时刻**：当前 `StateX_new` 不是语义严格的当前 GNSS 验前名义状态，需要与同时刻 H/R/Inno 重新严格对齐。
6. **训练与部署**：论文的最终结果是连续EKF闭环结果；当前项目不能只根据离线单步loss宣称整段闭环精度改善。

## 10. 对当前项目的训练解读

当前项目已经保留了论文最核心的训练思想：

```text
IMU+速度
   ↓
Set Transformer
   ↓
QAcc对角缩放系数
   ↓
可微协方差预测与Kalman更新
   ↓
ECEF位置MSE
   ↓
反向传播训练网络
```

数学上，用 `PhiAcc/PStart/QAcc` 表示一个观测间隔的高频协方差传播是成立的；前提是：

$$
\mathbf{P}_{pred,cpp}
\approx
\mathbf{\Phi}_{acc}\mathbf{P}_{start}\mathbf{\Phi}_{acc}^T
+\mathbf{Q}_{acc}.
$$

当前最需要解决的不是 Kalman-informed loss 公式本身，而是：

- 严格验前名义状态的输出；
- H/R/Inno 的迭代时刻一致性；
- 正式训练与小实验 Q 缩放函数的统一；
- 训练后网络回灌 C++ 的连续闭环验证。

## 11. 简要结论

A-KIT 不使用 Q 真值标签。它让网络输出 Q 缩放系数，再通过可微 EKF 将可观状态真值误差反向传播到 Q 和网络参数。训练时利用保存的 EKF 中间量重放一个观测间隔；部署时将网络接入连续 EKF 闭环。

本项目对论文方法的主要改造是：将 GNSS 间隔内的高频 `Phi/Q` 事先累计为 `PhiAcc/QAcc`，使 Kalman-informed loss 只执行一次区间级协方差预测。区间传播公式是精确的；但当前“缩放累计后 QAcc”的网络参数化与论文“逐预测步使用网络 Q”属于不同实现，不能直接宣称二者严格等价。数据时刻对齐和最终 C++ 连续闭环验证仍是必要条件。
