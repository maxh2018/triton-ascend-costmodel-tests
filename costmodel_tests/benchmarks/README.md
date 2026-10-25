# 通用 benchmark 驱动

本目录保存可复用的 CostModel benchmark 驱动和统一 profiler 辅助代码。

- 具体 shape、参数和 train/holdout 属性由 `cases/*/manifest.json` 定义。
- 默认 100 个泛化用例的唯一维护清单是 `cases/ci/manifest.json`。
- 仅针对某个 `StageModelKind` 的探索性用例放在 `cases/personal/`，不直接加入 CI 泛化清单。
- 编译缓存、profiler 原始数据和运行报告不提交到本目录。

benchmark 驱动不是独立的用例清单；请从仓库根目录通过
`python costmodel_tests/run.py ...` 运行。
