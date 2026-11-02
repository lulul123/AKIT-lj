议题7：AI增强定位的标准化构建与调用（卢洁）
进展：初步梳理了AI增强定位的模型构建、模型管理及IPS调用的标准化流程。
AI组件分为三类：1）直接监督训练类，把输入和输出标签按照规划构建好；2）间接监督训练类，在1的基础上，把中间黑盒程序放好，相关源码在自己电脑上存在，对应好；3）其它类，比如生成式AI模型；4）未来可能还有其它，按照规范都构建好
方案：（1）数据标准化：统一数据组织方式、数据格式与输入输出规范，暂定采用json文件管理数据的配置，后续开发数据检查skill。（2）模型管理：统一Python/PyTorch开发环境，规范模型代码、训练配置及权重的存储，建立统一模型接口。（3）模型调用：初期通过文件与IPS交互，后续采用进程间通信实现实时调用模型，实现推理预测，并设置超时回退机制，避免影响正常解算。（4）部署共享：逐步实现整体流程的标准化，并部署到服务器上。
以惯导过程噪声学习为试点，完成数据准备、模型训练、预测及IPS解算的标准化流程的构建。后续完成标准化流程在服务器上的部署，以及其他训练模型的接入,最终希望能够实现这一流程的AI自动调用与执行。

## 当前落地：模型管理 v1

- 标准训练配置：`configs/akit_set_transformer.json`；
- 配置约束：`schemas/training.schema.json`；
- 配置检查：`tools/validate_training_config.py`；
- 通用模型接口与注册表：`Code/akit_model_api.py`；
- 统一接口版本：`akit-model-1`；
- 输入：`imu (B,C,T)` 与 `p_start (B,N,N)`；
- 输出：`raw_qc_scale_logits (B,N)`，正值变换和卡尔曼链路由训练任务统一处理；
- 当前模型注册名：`set_transformer`。

配置化小规模训练：

```bash
python Code/A-KIT_lj_agent.py --config configs/akit_set_transformer.json --small --epochs 1
```

接入新模型时，实现相同的 `forward(imu, p_start)` 张量接口，通过
`register_model()` 注册，并在配置中替换 `model.name` 与 `model.parameters`。
