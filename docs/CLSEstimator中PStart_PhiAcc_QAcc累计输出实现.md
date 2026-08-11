# CLSEstimator 中 PStart、PhiAcc、QAcc 累计输出实现

## 1. 修改目的

训练阶段不再使用 GNSS 到达前最后一个 IMU 传播步的单步 `Phi` 和 `Q`，而是使用相邻两次成功 GNSS 位置更新之间的等效量：

$$
P_{pred}=\Phi_{acc}P_{start}\Phi_{acc}^{T}+Q_{acc}
$$

本次修改位于 `CLSEstimator.cpp` 的 `CGLSEstimator::KFPrediction()`，旧的 `Q.txt`、`Phi.txt` 和 `StateP.txt` 单步输出被替换为区间累计输出。

## 2. 输出文件

| 文件 | 内容 | 输出时机 |
|---|---|---|
| `PStart.txt` | 当前累计区间起点的协方差 | GNSS 位置更新通过质控后 |
| `PhiAcc.txt` | 区间内所有单步 Phi 的有序乘积 | 同上 |
| `QAcc.txt` | 区间内所有单步 Q 的等效累计结果 | 同上 |
| `PPredCpp.txt` | C++ 原始逐步递推得到的当前先验协方差 `StatePPre` | 同上 |
| `AccMeta.txt` | 时间戳和本区间实际累计的传播步数 | 同上 |

其中 `PPredCpp.txt` 和 `AccMeta.txt` 是验证文件，不是网络的必要输入。

## 3. 为什么不能在区间末端直接把传入的 StateP 当作 PStart

`KFPrediction()` 中原公式为：

```cpp
MatrixMultiply_HPHT(Phi.row, Phi.col, Phi.x, StateP.x, false, StatePPre.x);
StatePPre.add2(m_KFTmp_Q);
```

这里传入的 `StateP` 是当前单步传播之前的协方差。如果当前调用已经位于区间末端，直接输出这个 `StateP`，得到的是“最后一个 IMU 步之前的 P”，不是整个区间起点的 P。

因此实现中在新区间第一次进入 `KFPrediction()` 时保存：

```text
PStart = StateP
```

在后续传播步中 `PStart` 保持不变，直到下一次 GNSS 位置更新通过质控后输出。

成功更新输出完成时仅设置 `m_StartNewInterval = true`。下一次进入 `KFPrediction()` 时再读取传入的 `StateP`，这样可以取得本次 GNSS 量测处理后的后验协方差，作为下一区间起点。

## 4. 持久累计变量

累计量不能声明为普通局部变量，否则每次进入 `KFPrediction()` 都会被销毁。本次在函数内增加：

```cpp
Mat m_PStart;
Mat m_PhiAcc;
Mat m_QAcc;
bool m_AccInitialized;
bool m_StartNewInterval;
bool m_HasSuccessfulGNSSBoundary;
int m_AccStepCount;
```

这些变量已经改成 `CGLSEstimator` 成员，避免多个估计器实例共享累计状态。当前仓库只有 `referance/CLSEstimator.h` 副本；实际编译前仍需把同样的声明同步到真实的 `BASEDLL/CLSEstimator.h`。

## 5. PhiAcc 递推

新区间初始化：

$$
\Phi_{acc,0}=I
$$

每个传播步执行：

$$
\Phi_{acc,i}=\Phi_i\Phi_{acc,i-1}
$$

代码为：

```cpp
Mat PhiAccNew(PhiAcc.row, PhiAcc.col);
PhiAccNew.multiply(Phi, PhiAcc);
PhiAcc = PhiAccNew;
```

最终顺序为：

$$
\Phi_{acc}=\Phi_N\Phi_{N-1}\cdots\Phi_2\Phi_1
$$

不能写成 `PhiAcc * Phi`，因为矩阵乘法不可交换。

## 6. QAcc 递推

新区间初始化：

$$
Q_{acc,0}=0
$$

当前单步离散 Q 仍由原代码计算：

$$
Q_i=\frac{1}{2}(\Phi_iQ_{c,i}\Phi_i^T+Q_{c,i})\Delta t_i
$$

然后累计：

$$
Q_{acc,i}=\Phi_iQ_{acc,i-1}\Phi_i^T+Q_i
$$

代码为：

```cpp
Mat QAccNew(QAcc.row, QAcc.col);
MatrixMultiply_HPHT(Phi.row, Phi.col, Phi.x, QAcc.x, false, QAccNew.x);
QAccNew.add2(m_KFTmp_Q);
QAcc = QAccNew;
```

`QAcc` 不能用单步 Q 的直接求和代替，因为较早注入的噪声还会被后续 Phi 继续传播。

## 7. 每次调用的执行顺序

正 `dt` 的一次 `KFPrediction()` 依次执行：

1. 计算当前单步 `Phi`；
2. 计算当前单步 `m_KFTmp_Q`；
3. 必要时初始化新区间并保存 `PStart`；
4. 用当前 `Phi` 和 `m_KFTmp_Q` 更新 `PhiAcc/QAcc`；
5. 原代码正常计算 `StatePPre = Phi * StateP * Phi^T + Q`；
6. `runOneSubLC()` 完成质量控制；
7. 仅当 `bPassUPT && (ELCType & ELCTYPE_POS)` 时设置 `updateTime` 并调用 `FinishGNSSInterval()`；
8. 输出累计矩阵和 `SINSPPre`，标记下一次传播开始新区间。

因此成功 GNSS 更新之前的最后一个传播步已经包含在 `PhiAcc` 和 `QAcc` 中；更新失败时不会重置累计器。

## 8. updateTime 的来源

`CLooseCoupling.cpp` 在 GNSS 位置更新最终通过质控后执行：

```cpp
if (bPassUPT && (ELCType&ELCTYPE_POS)) {
    m_GLS.updateTime = m_UpdateTime.secsOfWeek;
    m_GLS.FinishGNSSInterval(m_GLSInfo.SINSPPre);
}
```

当前 GNSS 数据只在整数秒更新，因此 `int updateTime` 足以保存输出时间戳。构造函数中初始化：

```cpp
updateTime = 0;
```

避免读取未初始化的时间变量；输出完成后重新清零。

## 9. 必须进行的一致性验证

生成新数据后，在 Python 中按相同时间戳读取：

```text
PStart.txt
PhiAcc.txt
QAcc.txt
PPredCpp.txt
AccMeta.txt
```

重建：

```python
P_pred_rebuilt = Phi_acc @ P_start @ Phi_acc.T + Q_acc
```

并计算：

```python
error_max = np.max(np.abs(P_pred_rebuilt - P_pred_cpp))
error_rel = np.linalg.norm(P_pred_rebuilt - P_pred_cpp) / (
    np.linalg.norm(P_pred_cpp) + 1e-15
)
```

只有 `P_pred_rebuilt` 与 `PPredCpp` 在浮点误差范围内一致，才能确认：

- `PStart` 捕获时刻正确；
- Phi 乘法顺序正确；
- Q 递推正确；
- 成功 GNSS 更新前最后一个传播步没有遗漏；
- 区间重置没有提前或延后。

同时检查 `AccMeta.txt`。100 Hz IMU、稳定 1 Hz 区间通常应接近 100 步；插值、起始区间或数据缺失可能使步数不同。

## 10. 当前边界和后续需要确认的事项

### 10.1 当前按成功 GNSS 位置更新划分区间

本次实现由 `runOneSubLC()` 的最终质量控制结果触发。只有：

```cpp
bPassUPT && (ELCType & ELCTYPE_POS)
```

才结束区间。GNSS 缺失或位置更新被拒绝时继续累计到下一次成功更新，因此区间可能长于 1 秒。

### 10.2 第一个输出区间可能不足 1 秒

第一次成功 GNSS 更新只建立起始边界，不输出启动阶段的不完整区间。从第二次成功更新开始输出完整样本。

### 10.3 多实例

累计矩阵和控制标志已经是类成员，各实例独立维护区间状态。输出文件句柄仍为函数静态变量，因此多个实例若同时写同一组文件，仍需进一步拆分输出路径或文件句柄。

### 10.4 NominalState_prior

`PStart/PhiAcc/QAcc` 在 `CLSEstimator.cpp` 中产生；网络状态输入 `NominalState_prior.txt` 应在 `CLooseCoupling.cpp` 的 GNSS 更新前输出。它是绝对名义状态，不应使用更新后的 `StateX_new.txt` 代替。

## 11. 与 Python 训练代码的对应关系

Python 数据字段为：

```text
PStart.txt             -> sample["P_start"]
PhiAcc.txt             -> sample["Phi_acc"]
QAcc.txt               -> sample["Q_acc"]
NominalState_prior.txt -> sample["state_pred"]
```

网络预测累计 Q 的缩放系数后只进行一次区间级协方差预测：

```python
P_base = Phi_acc @ P_start @ Phi_acc.transpose(-2, -1)
P_pred = P_base + Q_acc_pred
```

不能再次进行高频 Q 递推，也不能再对 `state_pred` 乘一次 `Phi_acc`。
