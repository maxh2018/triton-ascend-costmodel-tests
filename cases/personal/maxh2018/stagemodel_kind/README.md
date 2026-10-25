# StageModelKind 专项用例

这里保存针对特定 `StageModelKind`、公式边界或校准假设的个人实验清单。
这些用例用于研究和定向复测，不自动进入 `cases/ci/manifest.json` 的泛化门禁。

当前两个 manifest 是 29 个 structural holdout 的历史分组；它们复用
`costmodel_tests/benchmarks/` 下的 `bench_pytest_ut_structural_costmodel.py`
和 `bench_pytest_ut_more_costmodel.py`。

在仓库根目录显式运行：

```bash
python costmodel_tests/run.py \
  --manifest cases/personal/maxh2018/stagemodel_kind/pytest_ut_structural_17_manifest.json \
  --device 0
```

新增专项用例时应保留独立 manifest，并说明对应的 `StageModelKind`、验证假设和适用范围。
只有证明具有跨算子、跨 shape 通用性后，才将相应配置纳入 CI 泛化清单。
