A-KIT schemas 目录说明
======================

1. datasets.schema.json

用途：
  定义 data/datasets.json 的 JSON Schema，用于约束项目级数据目录的基本结构。

当前检查内容：
  - 项目、默认配置、数据划分和数据集列表等必需字段；
  - dataset_id、trajectory_id 和 split 等字段格式；
  - GPS 周、路径、参考位置文件和验证状态；
  - 公共文件映射、样本契约和必需文件键的基本类型。

设计原则：
  - 只约束当前流程实际使用的字段；
  - 尚未确认的时间语义、坐标系和单位不写成强制规则；
  - 与 data/datasets.json 的 schema_version 配套维护；
  - 新增或删除必需配置字段时，应同步修改本 Schema。

当前版本：
  akit-dataset-catalog-0.1-draft

2. training.schema.json

用途：
  约束可复用训练配置，包括数据划分、模型接口版本、模型参数、优化器、
  损失和 checkpoint 策略。对应示例为 configs/akit_set_transformer.json。

备注：
  当前项目环境未安装 jsonschema 包。tools/validate_dataset.py 内置了必要的
  配置结构检查，因此运行数据检查器不依赖额外库。本文件仍采用标准的
  JSON Schema Draft 2020-12，后续可接入任意兼容的 Schema 校验工具。
