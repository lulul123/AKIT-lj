# A-KIT 误差状态 Kalman 流程说明

## 1. 先回答当前最容易混淆的问题

不能采用下面的流程：

```text
取 StateX_new 第一个值作为绝对状态
                ↓
后续直接使用 stateX.txt 继续推算绝对位置
```

原因是两份文件表示不同的数学对象：

```text
StateX_new.txt = 名义/绝对导航状态 x
stateX.txt     = 误差状态或误差状态 Kalman 输出 δx
```

`stateX.txt` 中的位置分量不是下一时刻的 ECEF 绝对坐标，而是对当前名义位置的误差估计或修正量。它必须按照原程序规定的符号反馈到名义状态，不能替代名义状态。

正确思路是：

```text
StateX_new 的第一个有效值
        ↓ 作为名义状态初值
100 Hz IMU惯导机械编排
        ↓ 产生下一时刻名义位置、速度、姿态
误差状态Kalman滤波
        ↓ 估计名义状态误差
误差反馈
        ↓ 修正名义位置、速度、姿态和零偏
修正后的名义状态继续进入下一时刻
```

因此，“只保存一个 `StateX_new` 初值”原则上可以，但前提是 Python 已经完整实现与原程序一致的惯导机械编排、误差反馈和协方差递推。没有机械编排，仅靠 `stateX.txt` 无法恢复后续绝对轨迹。

---

## 2. 当前15维状态中实际存在的两个向量

### 2.1 名义导航状态

当前确认 `StateX_new.txt` 的排列是：

| 索引 | 名义状态 | 当前已知信息 |
|---:|---|---|
| 0–2 | ECEF绝对位置 | 单位为m |
| 3–5 | 速度 | 坐标系和单位仍需从原程序确认 |
| 6–8 | 姿态 | 参数化、旋转顺序和单位仍需确认 |
| 9–11 | 加速度计零偏估计 | `StateX_new` 生成代码实际先写 `AccelBias`；单位和轴系仍需确认 |
| 12–14 | 陀螺零偏估计 | `StateX_new` 生成代码随后写 `GyrosBias`；单位和轴系仍需确认 |

记作：

$$
x=
\begin{bmatrix}
r^e & v & \Theta & b_a & b_g
\end{bmatrix}^T
$$

它是最终需要输出的导航解。loss 使用其中反馈更新后的 ECEF 位置与参考位置比较。

### 2.2 误差状态

`stateX.txt` 是误差状态 Kalman 输出。若其排列与名义状态分组一致，可记作：

$$
\delta x=
\begin{bmatrix}
\delta r & \delta v & \delta\theta & \delta b_a & \delta b_g
\end{bmatrix}^T
$$

其中：

- $\delta r$ 是位置误差或位置修正量，不是ECEF绝对位置；
- $\delta v$ 是速度误差；
- $\delta\theta$ 是小角度姿态误差，不是绝对姿态角；
- $\delta b_g$、$\delta b_a$ 是零偏估计误差或修正量。

仍必须核对原程序的误差定义：

$$
\delta x=x_{true}-x_{nominal}
$$

还是：

$$
\delta x=x_{nominal}-x_{true}
$$

这决定反馈时使用加号还是减号。

---

## 3. 三条不能混在一起的计算链路

### 3.1 名义状态传播链路

名义状态由IMU机械编排传播：

$$
x_k^-=f(x_{k-1}^+,u_k,\Delta t)
$$

这里的 $u_k$ 包含陀螺和加速度计测量。实际计算需要包括：

1. 使用当前零偏修正陀螺和加速度计；
2. 更新姿态；
3. 将比力转换到导航坐标系或ECEF；
4. 计算重力、地球自转和科里奥利项；
5. 更新速度；
6. 更新位置。

它通常是非线性函数，不能简单写成：

```python
absolute_state_next = Phi @ absolute_state
```

### 3.2 误差协方差传播链路

误差动力学线性化后得到：

$$
\dot{\delta x}=F\delta x+Gw
$$

离散化后：

$$
\Phi_k=\exp(F_k\Delta t)
$$

协方差传播为：

$$
P_k^-=\Phi_kP_{k-1}^+\Phi_k^T+Q_k
$$

`Phi.txt`、`StateP.txt` 和 `Q.txt` 大概率属于这条误差状态链路。必须从原程序确认它们的时刻和频率。

### 3.3 GNSS量测更新与误差反馈链路

GNSS到达时，先根据GNSS位置和惯导名义位置构造新息。例如：

$$
z_k=r_{GNSS,k}-r_{INS,k}^-
$$

误差状态量测模型为：

$$
z_k=H_k\delta x_k+v_k
$$

然后计算：

$$
S_k=H_kP_k^-H_k^T+R_k
$$

$$
K_k=P_k^-H_k^TS_k^{-1}
$$

$$
\widehat{\delta x}_k=K_kz_k
$$

最后将误差估计反馈到名义状态：

$$
x_k^+=x_k^-\boxplus\widehat{\delta x}_k
$$

$\boxplus$ 表示按照状态类型进行反馈：位置和速度通常是加减修正；姿态通常需要小角度旋转反馈，不能未经确认直接把三个姿态分量相加。

反馈完成后，误差状态通常重置为零：

$$
\delta x_k^+\leftarrow0
$$

协方差还可能需要误差重置 Jacobian：

$$
P_k^+\leftarrow G_{reset}P_k^+G_{reset}^T
$$

---

## 4. 你的数据文件应放在流程的什么位置

以下映射是当前最合理的解释，但“先验/后验”和具体时间标签仍需用原程序核对。

| 文件 | 在正确流程中的作用 | 能否每秒直接覆盖递推状态 |
|---|---|:---:|
| `imu.txt` | 100 Hz名义状态机械编排输入，同时影响误差模型 | 否 |
| `Vel.txt` | 当前网络输入或辅助观测；具体来源和物理作用需确认 | 否 |
| `StateX_new.txt` | 名义绝对状态的初始化值和逐时刻对照真值/基线 | 正式递推中不能每秒覆盖 |
| `stateX.txt` | 原程序输出的误差状态，用于审计误差反馈 | 不能作为绝对状态 |
| `Phi.txt` | 误差状态离散转移矩阵 | 只传播误差/协方差 |
| `StateP.txt` | 误差状态协方差 | 是递推量，但须先确认是先验还是后验 |
| `Q.txt` | 误差状态过程噪声 | 加入协方差传播 |
| `H.txt` | 误差状态量测矩阵 | 用于GNSS量测更新 |
| `R.txt` | GNSS量测噪声协方差 | 用于GNSS量测更新 |
| `Inno.txt` | 原程序生成的新息 | 可用于复现审计；正式运行也可由GNSS和INS重新计算 |
| `.imr.ref`/`.pos` | 参考绝对位置 | 用于loss和最终评价，不进入状态传播 |

`StateX_new.txt` 的逐时刻记录在开发阶段非常有价值：它可以作为原程序输出，与Python递推结果逐时刻比较。但正式训练时，如果每一秒都从文件重新读取并覆盖Python的当前状态，就会切断连续递推，使前一秒网络预测的Q无法影响后一秒状态。

---

## 5. 从第一条数据开始的正确时间流程

假设每条轨迹的初始时间为 $t_0$，IMU为100 Hz，GNSS为1 Hz。

### 5.1 初始化

从原程序导出或读取：

```text
nominal_state = StateX_new(t0)
P_post        = P0或确认后的StateP(t0)
delta_x       = 0
```

这里可以使用 `StateX_new` 的第一个有效值，但必须同时取得对应的初始协方差和精确时间定义。

### 5.2 一秒内的100次传播

对每个IMU采样执行：

```python
nominal_state = mechanize_nominal_state(
    nominal_state,
    gyro_measurement,
    accel_measurement,
    dt=0.01,
)

Phi_i, Q_i = build_or_load_error_dynamics(...)
P = Phi_i @ P @ Phi_i.T + Q_i
```

如果现有 `Phi.txt` 和 `Q.txt` 只按1 Hz保存，则必须确认它们是否已经严格累计了这一秒内100步的传播。不能把一个0.01秒矩阵只执行一次，也不能把一个一秒累计矩阵执行100次。

### 5.3 GNSS到达时更新

```python
innovation = build_innovation(gnss_position, nominal_state.position)
K = P @ H.T @ inverse(H @ P @ H.T + R)
delta_x = K @ innovation

nominal_state = inject_error(nominal_state, delta_x)
P = joseph_covariance_update(P, K, H, R)
P = reset_covariance_if_required(P, delta_x)
delta_x = zeros(15)
```

更新后的 `nominal_state` 和 `P` 是下一段100 Hz IMU传播的初值，而不是直接跨过一秒。它们首先进入下一个0.01秒传播步；连续处理这一秒内的100个IMU样本后得到下一整秒的先验状态和先验协方差，再执行下一次1 Hz GNSS量测更新。

```text
x_post(k), P_post(k)
        ↓ 作为初值
100次IMU/协方差传播
        ↓
x_prior(k+1), P_prior(k+1)
        ↓ GNSS量测更新和误差反馈
x_post(k+1), P_post(k+1)
```

名义状态原则上按100 Hz机械编排。协方差有两种等效实现：使用0.01秒 `Phi_i/Q_i` 连续传播100次，或使用严格累计得到的一秒 `PhiBar/QBar` 传播一次。只有在确认现有文件属于后一种情况时，才能对协方差采用1 Hz累计传播。

### 5.4 计算训练loss

使用反馈后的名义ECEF位置：

$$
L_k=\left\|r_{nominal,k}^+-r_{reference,k}^{ECEF}\right\|_2^2
$$

为避免数百万米ECEF坐标造成数值精度问题，应使用 `float64`，并直接计算位置差。也可以将位置残差转换到局部ENU/NED后计算等价的米制误差，但转换方式必须对所有模型保持一致。

---

## 6. A-KIT网络预测的Q应该放在哪里

网络不负责直接预测绝对位置，也不直接替代惯导机械编排。它预测误差状态过程噪声，例如第一版采用对标称Q的正缩放：

$$
Q_{network,k}=D(s_k)Q_{nominal,k}D(s_k)
$$

或采用当前代码更接近的对角缩放形式。无论选择哪一种，必须保证 $Q_{network}$ 对称半正定，并且训练、验证、测试使用完全相同的变换。

网络Q进入：

$$
P_k^-=\Phi_kP_{k-1}^+\Phi_k^T+Q_{network,k}
$$

它通过以下链路影响loss：

```text
网络输出Q
   ↓
改变P_prior
   ↓
改变Kalman gain
   ↓
改变误差状态修正量
   ↓
改变反馈后的绝对位置
   ↓
改变位置loss
```

这条链路必须保持PyTorch可微，同时也必须保持真实Kalman时序连续。

---

## 7. 当前Python代码为什么只能算诊断版本

当前 `akit_utils_agent.kalman_filter()` 的核心状态计算是：

```python
state_pred = Phi @ StateX_new
state_post = state_pred + K @ Inno
```

这里混合了误差状态矩阵和绝对名义状态。若 `Phi` 是误差状态转移矩阵，则第一行不成立；第二行也遗漏了姿态误差注入、反馈符号和误差重置。

当前数据加载还在每个秒级样本中重新读取 `StateX_new` 和 `StateP`，不同样本彼此独立，因此上一秒网络Q产生的后验状态和协方差不会进入下一秒。这可以用于检查网络、梯度和文件读取，但不能作为正式连续组合导航结果。

正式版本至少需要拆成以下接口：

```python
nominal_state = mechanize_nominal_state(nominal_state, imu_sample, dt)
Phi, Q = error_state_dynamics(nominal_state, imu_sample, network_output)
P_prior = propagate_covariance(P_post, Phi, Q)
delta_x, P_post = measurement_update(P_prior, H, R, innovation)
nominal_state = inject_and_reset(nominal_state, delta_x)
```

### 7.1 已确认的当前数据模式：外部连续滤波、Python单步重计算

当前数据中的状态和矩阵已确认由外部组合导航系统连续运行后逐时刻导出。因此，Python不一定要在第一版训练中立即重写整套连续机械编排；可以先进行严格定义的单步局部Q学习。

单步方法成立的关键不是执行 `Phi @ StateX_new`，而是把外部系统在当前GNSS时刻的名义先验状态视为固定工作点，只让网络Q改变当前先验协方差、Kalman增益和误差反馈：

```python
# 均来自同一个GNSS更新时刻，并明确先验/后验语义
nominal_prior = exported_nominal_prior
P_base = Phi @ previous_P_post @ Phi.T

Q_theta = network(imu_window, ...)
P_prior = P_base + Q_theta
K = kalman_gain(P_prior, H, R)
delta_x = K @ innovation
nominal_post = inject_error(nominal_prior, delta_x)
loss = position_loss(nominal_post.position, reference_position)
```

这种方法学习的是：“在外部系统给定的当前工作点上，怎样调整Q可以改善当前GNSS更新后的绝对位置。”它不学习网络Q经历史反馈后对未来状态、协方差和新息的累计影响。因此它适合作为单步基线和Q可学习性验证，但其结果不能直接宣称为连续自适应Kalman滤波性能。

#### 单步方法成立必须同时满足

1. `nominal_prior` 是当前量测更新前的名义绝对状态，而不是已经使用原始Q更新后的状态；
2. 输入协方差能够表示 $\Phi P_{previous,post}\Phi^T$，网络Q在此基础上只加入一次；
3. `innovation/H/R` 与该名义先验状态和同一GNSS时刻严格对应；
4. 网络预测的Q对应从上一个更新时刻到当前更新时刻的传播区间；
5. loss参考位置独立于原滤波输出，不能把原系统更新后的 `StateX_new` 同时当输入和监督标签；
6. 训练、验证和未来部署使用相同的Q参数化与误差反馈符号。

#### 现有导出不满足时必须补充

至少应从外部组合导航系统逐GNSS时刻导出：

```text
nominal_prior      更新前名义状态
nominal_post       原系统更新后名义状态，仅用于对照
P_previous_post    上一更新时刻后验协方差
P_current_prior    当前更新时刻原系统先验协方差，用于核验
P_current_post     当前更新时刻原系统后验协方差，用于核验
PhiBar             两次GNSS更新之间的累计误差转移矩阵
QBar_original      同一传播区间的累计原始过程噪声
innovation, H, R   当前量测更新数据
reference_position 独立参考绝对位置
```

如果只能取得已经更新后的 `StateX_new` 和语义不明确的单个 `StateP`，就无法保证单步反事实Q替换在数学上正确，应先补充导出而不是继续猜测。

---

## 8. 在继续改代码前必须确认或补充的数据

按优先级排序：

1. 找到原组合导航程序中生成 `Phi.txt`、`StateP.txt`、`Inno.txt`、`stateX.txt` 和 `StateX_new.txt` 的代码位置。
2. 确认误差定义及反馈正负号。
3. 确认 `StateX_new` 的速度坐标系、姿态表示、两类零偏单位。
4. 确认各文件在同一时间标签下分别表示更新前还是更新后。
5. 确认 `Phi/Q` 的频率：100 Hz单步量还是1 Hz累计量。
6. 导出每条轨迹的初始名义状态、初始协方差和起始时间。
7. 若Python无法从现有IMU数据复现机械编排，补充原程序的重力模型、地球参数、坐标转换、IMU增量/速率定义和补偿方式。

在这些信息确认前，不应凭常见公式猜测姿态反馈或符号，因为一个符号、坐标系或时间错位就可能导致轨迹发散。

---

## 9. 推荐实施顺序与验收标准

### 阶段A：固定Q复现原程序

先完全关闭网络，使用原始Q：

```text
Python初始状态 = 原程序初始状态
Python初始P    = 原程序初始P
Python IMU传播 = 原程序机械编排
Python Q       = 原始固定Q
```

逐时刻比较：

- 名义位置、速度、姿态和零偏；
- `P_prior`、Kalman gain、误差状态更新；
- 反馈后的名义状态和 `P_post`。

只有固定Q能够复现原程序，才能证明Kalman时序、坐标系和反馈公式正确。

### 阶段B：网络恒等输出复现

让网络始终输出Q缩放1，要求结果与阶段A一致。这一步验证网络Q接口没有改变基线算法。

### 阶段C：短窗口可微训练

按连续时间窗口训练，让前一秒的后验状态和P进入下一秒。先使用较短窗口，并在窗口边界截断梯度。

### 阶段D：按完整轨迹评价

```text
训练：data065517、data090854、data122041
验证：data212832
最终测试：data013325
```

`data013325` 在模型结构、超参数和checkpoint选择全部结束前不得参与训练或调参。

---

## 10. 一句话记忆

```text
StateX_new给出“当前导航解在哪里”；
IMU机械编排负责“下一步走到哪里”；
误差状态Kalman负责“这个推算错了多少”；
误差反馈负责“用估计误差修正导航解”；
网络Q负责“调节对模型不确定性的信任程度”。
```

## 11. `组合数据提取.pdf` 输出位置审计

根据随后补充的完整 `CLooseCoupling.cpp` 和 `CLSEstimator.cpp`，当前各文件确实不是在同一个统一的GNSS更新切面上导出，不能只按相同整数秒直接假定它们具有一致的先验/后验语义。

### 11.1 完整源码确认的调用顺序

`runOneLC()` 的实际顺序是：

```text
PrepareLC()
  └─ KFPre()
       ├─ 100 Hz IMU机械编排
       └─ 当前dt协方差预测
runOneSubLC()
  └─ GLS量测更新
UpdateSolution()
  └─ 误差反馈到绝对导航解并输出StateX_new
UpdateClosedLoop()
  └─ 将已反馈的误差状态分量清零，P保持不变
```

具体为：

1. `KFPre()` 首先执行 `runOneMechanization()`，说明绝对位置、速度和姿态由IMU机械编排推进；
2. 随后根据当前 `dt` 计算 `Phi = exp(CF * dt)`，离散化当前步Q，并计算 `SINSPPre = Phi * StateP * Phi^T + Q`；
3. `runOneSubLC()` 将 `m_GLSInfo.SINSPPre` 赋给 `m_GLS.m_StatePp`，因此GLS使用的是当前预测后的先验协方差；
4. `H/R/Inno` 的输出位置位于 `m_GLS.GLS()` 调用之前，因此它们是当前量测更新输入；
5. `m_GLS.GLS()` 内部明确执行 `StateX = StateXp + K * Inno` 和Joseph后验协方差更新；
6. `m_GLS.GLS()` 完成后，代码执行：

```cpp
m_GLSInfo.StateX = m_GLS.m_StateX;
m_GLSInfo.StateP = m_GLS.m_StateP;
```

因此这里回写的是量测更新后的误差状态和后验协方差；
7. `UpdateSolution_GLSInfo()` 使用机械编排结果减去位置/速度误差，进行姿态小角度修正，并反馈零偏；
8. `StateX_new` 随后从 `Position/Velocity/Attitude/AccelBias/GyrosBias` 组装，因此它是误差反馈后的绝对后验导航解；
9. `UpdateClosedLoop()` 最后把已反馈的位置、速度、姿态及零偏误差状态清零，但协方差保持后验值并进入后续传播；
10. `StateX_new` 和 `stateX` 输出代码均使用 `m_UpdateTime.secsOfWeek + 1`，而其他矩阵使用 `m_UpdateTime.secsOfWeek`，存在明确的1秒标签偏移。

此外，`StateX_new` 的实际输出顺序是：

```text
0–2   Position
3–5   Velocity
6–8   Attitude
9–11  AccelBias
12–14 GyrosBias
```

这与先前口述的“9–11陀螺零偏、12–14加计零偏”相反，必须按实际生成代码修正Python、文档和Q各维含义。

### 11.2 `Phi/Q` 当前不能视为一秒累计量

截图中的预测代码为：

```cpp
m_KFTmp_F = CF;
m_KFTmp_F *= dt;
Phi.exp(m_KFTmp_F, 2);

// 当前dt的Q离散化
Q = 0.5 * (Phi * Qc * Phi_T + Qc) * dt;

// 当前传播步
StatePPre = Phi * StateP * Phi_T + Q;
```

这说明该函数里的 `Phi/Q` 是当前一次 `dt` 的传播量。PDF截图没有显示将一秒内100个 `Phi_i/Q_i` 累计为 `PhiBar/QBar` 的代码。因此，除非外部还有未截取的累计逻辑，当前导出的单个 `Phi.txt/Q.txt` 不能直接解释为两次1 Hz GNSS更新之间的累计传播量。

### 11.3 `StateP/Phi/Q` 的实际错位

完整源码显示，`StateP/Phi` 在当前 `KFPrediction()` 调用之前输出：

- `StateP.txt` 保存的是进入当前0.01秒预测步之前的协方差；它可能已经历此前多个100 Hz传播步，并不固定等于上一GNSS时刻的 $P_{post}$；
- `Phi.txt` 此时的 `SINSPhi` 尚未被当前调用更新，所以保存的是上一个0.01秒预测步留下的Phi；
- `Q.txt` 在当前 `KFPrediction()` 内计算，并通过整数秒设置的 `updateTime` 只输出当前0.01秒预测步的Q；
- `H/R/Inno` 则在完成当前预测之后、GLS量测更新之前输出。

因此同一个文件标签下实际混合了：预测前P、上一预测步Phi、当前预测步Q、当前量测更新输入，以及被标记为下一秒的后验绝对状态。当前Python按整数秒求交集不能修复这种错位。

`H/R/Inno/stateX` 的输出代码还位于最多7次的量测迭代循环内部，外层又按 `CUP/ZUP/POS/DMI|NHC` 尝试不同观测类型。因此同一个时间标签可能写入多组记录。当前Python矩阵读取器以时间为字典键，重复时间会由后写记录覆盖先写记录；最终保留的可能是最后一次迭代或另一种观测，而不一定是第一次位置GNSS更新。重新导出时必须限定明确的观测类型与迭代编号，并保证每个时间只写一个训练样本。

### 11.4 对现有单步训练的结论

现有单步训练目前不能被认定为数学上正确，原因包括：

1. 使用误差状态 `Phi` 直接传播反馈后的绝对状态 `StateX_new`；
2. `StateX_new` 很可能是后验绝对状态，再执行一次 `K*Inno` 会重复量测更新；
3. `StateX_new` 的时间标签人为加1秒，可能与其他矩阵错位；
4. 当前 `Phi/Q` 看起来是单个 `dt` 的传播量，而不是一秒累计量；
5. `StateP` 与当前 `H/R/Inno` 的先验/后验对应关系尚未统一。

这不意味着单步局部Q训练方案本身不可用，而是当前导出切面不足以支持可靠的Q反事实替换。

### 11.5 建议在同一个GNSS更新函数中重新导出

应在 `m_GLS.GLS()` 调用前后设置统一时间戳，一次性导出：

```text
time                    当前GNSS量测真实时间，不做隐式+1
measurement_type        明确限定POS等观测类型
iteration_index         明确选择用于训练的一次迭代，建议固定第一有效次或最终接受次
nominal_prior           量测更新前、误差反馈前的绝对导航状态
P_prior_original        原系统当前量测更新前协方差
P_base_without_Q        若可直接取得，导出PhiBar*P_previous_post*PhiBar^T
PhiBar                  上一GNSS更新到当前更新之间的累计转移矩阵
QBar_original           同一时间区间累计过程噪声
H, R, innovation        当前GNSS量测更新输入
delta_x_post_original   原系统量测更新得到的误差状态
P_post_original         原系统量测更新后的协方差
nominal_post_original   误差反馈后的绝对导航状态
reference_position      独立参考绝对位置
```

重新导出后，Python单步训练应固定 `nominal_prior/H/R/innovation`，使用网络替换 `QBar_original`，重新计算 `P_prior/K/delta_x/nominal_post/loss`。原系统的 `nominal_post_original` 只用于复现核对，不能同时作为网络更新前状态。

## 12. 不保存100组矩阵的一秒累计实现

GNSS为1 Hz并不表示GNSS之间停止运行。每个0.01秒IMU时刻都必须执行：

```text
名义位置/速度/姿态的INS机械编排
误差协方差传播（或为等效累计量更新递推器）
```

GNSS只在整秒提供量测更新。两次GNSS之间若没有其他量测，可以不保存100组 `Phi_i/Q_i`，而是在线累计两个15×15矩阵。

在上一GNSS更新完成后初始化：

$$
\bar\Phi_0=I,\qquad \bar Q_0=0
$$

每个IMU传播步计算当前 `Phi_i/Q_i` 后更新：

$$
\bar\Phi_i=\Phi_i\bar\Phi_{i-1}
$$

$$
\bar Q_i=\Phi_i\bar Q_{i-1}\Phi_i^T+Q_i
$$

第100步结束后有：

$$
P_{k+1}^-=\bar\Phi_{100}P_k^+\bar\Phi_{100}^T+\bar Q_{100}
$$

每秒只需导出：

```text
P_previous_post
PhiBar
QBar_original
nominal_prior_at_current_gnss
H, R, innovation
```

这与逐步执行100次协方差传播数学等价，但不需要保存100个文件块。名义INS状态仍然必须实际进行100次机械编排，不能用 `PhiBar` 代替。

### 12.1 网络每秒输出一个Q缩放时的两种方式

#### 方式一：上一秒IMU窗口预测下一秒Q（最容易实时部署）

在第 $k$ 秒GNSS更新时，网络使用此前一秒IMU窗口输出缩放 $s_k$，随后在 $k\rightarrow k+1$ 的100次预测中保持该缩放：

```text
k秒：网络读取过去100个IMU样本，输出s_k
      ↓
未来100个KFPrediction：Q_i_new = transform(Q_i_original, s_k)
      ↓
k+1秒GNSS更新并计算loss
```

这是严格因果、实现最直接的方案，但网络输出对loss有一秒延迟。

#### 方式二：当前一秒窗口在GNSS到达时预测本秒累计Q

如果两次GNSS之间没有ZUP、DMI等中间量测，名义机械编排与Q无关，Q只影响整秒末的先验协方差。此时可以一边运行INS，一边累计 `PhiBar` 和Q的组成项；到GNSS时网络读取刚结束的一秒IMU窗口，再重构本秒 `QBar_network` 并执行量测更新。

第一版若网络只预测一个全局正缩放 $s$，且每个单步Q都整体缩放，则：

$$
Q_{i,network}=sQ_{i,original}
$$

由于协方差累计对Q是线性的：

$$
\bar Q_{network}=s\bar Q_{original}
$$

因此只保存一个 `QBar_original` 即可。这是最简单、最容易验证的正式基线。

若网络预测15个独立对角缩放，通常不能直接写成 `D * QBar_original * D`，因为中间的Phi会混合各状态维。可以为每个Q对角分量在线累计一个基矩阵 $B_j$：

$$
B_{j,i}=\Phi_iB_{j,i-1}\Phi_i^T+Q_i^{(j)}
$$

GNSS时刻再组合：

$$
\bar Q_{network}=\sum_{j=1}^{15}s_jB_j
$$

这需要在线维护15个15×15矩阵，但仍远小于把每秒100组矩阵写入文件，也能保持15维缩放对网络可微。

### 12.2 适用限制

上述“一秒末重构QBar”只在两次GNSS之间没有其他Kalman量测更新时严格成立。如果区间内执行了ZUP、CUP、DMI、NHC等更新，Q会在中间时刻改变Kalman增益和后续P，此时必须按真实事件顺序逐步回放，不能只在整秒末替换一个QBar。

## 参考资料

1. Joan Solà, *Quaternion Kinematics for the Error-State Kalman Filter*: https://www.iri.upc.edu/people/jsola/publications/kinematics.pdf
2. Yarong Luo et al., *SE₂(3) based Extended Kalman Filtering and Smoothing Framework for Inertial-Integrated Navigation*: https://arxiv.org/abs/2102.12897
3. NASA Technical Reports Server, *State Transition Matrix for Inertial Navigation Systems*: https://ntrs.nasa.gov/citations/19700008298
4. MathWorks, `insfilterErrorState`: https://www.mathworks.com/help/nav/ref/insfiltererrorstate.html
