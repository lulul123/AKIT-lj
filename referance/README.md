# Reference source files

该目录保存从完整 IPS/POSMind 工程补充的只读参考源码，用于核对调用频率、类接口和 Kalman 变量定义，不是当前仓库的独立可编译源码树。

当前文件：

- `BaseSDC.h`：基础常量、宏和通用类型；
- `CLSEstimator.h`：GLS/Kalman 类接口和成员定义；
- `CSINSApplication.cpp`：应用主循环及 `runOneLC()` 调用时序；
- `CSINSMechanization.h`：机械编排接口和成员声明，缺少对应 `.cpp` 实现。

根目录的 `CLooseCoupling.cpp`、`CLSEstimator.cpp` 是当前正在修改的数据导出实现，因此不放入本目录。
