A-KIT data 目录说明
===================

1. datasets.json

用途：
  A-KIT 数据标准化流程的项目级唯一配置来源，集中管理公共数据契约、
  轨迹清单、文件映射和 train/validation/test 划分。

主要结构：
  project              项目名称、任务类型和当前处理流程
  metadata_resolution  配置合并和相对路径规则
  defaults             所有轨迹共用的时间、文件和样本定义
  split_policy         轨迹级数据划分规则
  datasets             每条轨迹的 ID、路径、GPS 周、参考文件和划分

目录关系：
  data/datasets.json
  data/data013325/
  data/data065517/
  data/data090854/
  data/data122041/
  data/data212832/

维护规则：
  - 公共文件名和样本形状只在 defaults 中维护一次；
  - 单条轨迹的差异写在 datasets 中对应条目内；
  - 某条轨迹有特殊配置时，在其条目中增加 overrides；
  - 不再在各轨迹目录中维护低信息量的 dataset.json；
  - 数据划分只在本文件中修改，避免出现多个配置来源；
  - 修改结构时同步更新 schemas/datasets.schema.json；
  - 新增轨迹后应运行 tools/validate_dataset.py 进行检查。

当前暂定划分：
  train       data065517、data090854、data122041
  validation  data212832
  test        data013325

注意：
  当前 schema_version 带有 draft，表示部分单位、坐标系和时间语义仍待确认。
  待确认事项统一记录在 docs/todolist.md 中。

PPredCpp 数值回放测试：
  python tools/validate_ppred_replay.py --samples 20

  测试按轨迹均匀抽取时刻，从 PStart 开始逐条回放 Phi 和 Qc，结果与
  C++ 输出的 PPredCpp 比较。--samples 0 会检查全部公共时刻（读取和计算
  时间较长）；也可在命令末尾指定一个或多个轨迹目录。
