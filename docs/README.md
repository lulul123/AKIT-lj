# 文档目录

当前研究路线固定为：在 C++ 中累计一个 GNSS/整数秒区间的等效
`PStart/PhiAcc/QAcc`，网络直接调整整段 `QAcc`，Python 只执行一次区间级
协方差预测，避免重复高频递推。

- [GNSS间隔等效Phi与Q累计方案](GNSS间隔等效Phi与Q累计方案.md)：当前方案的完整数学推导、数据语义、Python 使用方式和一致性验证方法；
- [CLSEstimator中PStart_PhiAcc_QAcc累计输出实现](CLSEstimator中PStart_PhiAcc_QAcc累计输出实现.md)：`CLSEstimator.cpp` 中当前累计器、输出时序、边界条件和后续确认事项；
- [运行与复现实验说明](运行与复现实验说明.md)：Python 环境、冒烟测试、训练命令和结果目录。

当前修改涉及仓库根目录的 `CLSEstimator.cpp`，Python 主路径位于 `Code/`。
首批新 C++ 数据生成后，必须先验证：

$$
P_{pred,cpp}\approx\Phi_{acc}P_{start}\Phi_{acc}^{T}+Q_{acc}
$$

验证通过后再进行正式训练。
