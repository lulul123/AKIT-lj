A-KIT tools 目录说明
====================

1. validate_dataset.py

用途：
  根据 data/datasets.json 检查 A-KIT 轨迹数据的文件完整性和基础格式。

当前检查内容：
  - 必需文件是否存在；
  - 数值是否包含 NaN 或 Inf；
  - 时间戳是否倒退或重复；
  - IMU、速度和 GNSSMech 的列数；
  - Phi、PStart、H、R 和 Inno 的矩阵形状；
  - Phi 的 dt 是否为正；
  - Qc 是否为完整的 15x15 矩阵；
  - 参考位置文件的 GPS 周是否与配置一致；
  - 文件时间范围和已检查记录数。

快速检查单条轨迹：
  .venv/bin/python tools/validate_dataset.py --dataset data065517

快速检查全部轨迹：
  .venv/bin/python tools/validate_dataset.py --all

全量检查单条轨迹并保存报告：
  .venv/bin/python tools/validate_dataset.py \
    --dataset data065517 \
    --max-records 0 \
    --report reports/data065517.json

参数说明：
  --catalog       项目级数据配置，默认 data/datasets.json
  --dataset       要检查的 dataset_id，可重复指定
  --all           检查配置中的全部轨迹
  --max-records   每个大文件最多检查的记录数，默认1000，0表示全量
  --report        可选的 JSON 报告输出路径

退出码：
  0  所选数据在本次扫描范围内通过
  1  至少一个数据集检查失败
  2  配置、参数或程序执行错误

注意：
  PASS (partial scan) 只表示抽查范围内格式正确，不代表全量数据已通过。
  当前版本尚不判断 PStart/Phi/GNSS 的真实时间语义，也不代替 todolist.md
  中要求的人工确认。

2. validate_training_config.py

用途：检查标准训练配置及模型输入输出契约字段。

运行：
  python tools/validate_training_config.py configs/akit_set_transformer.json

3. check_server_ready.py

用途：服务器克隆后的只读环境验收，检查代码、依赖、CUDA 和数据目录。

代码克隆后：
  python tools/check_server_ready.py --skip-data

数据传输后：
  python tools/check_server_ready.py
