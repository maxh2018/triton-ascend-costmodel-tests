# CostModel 回归测试入口

本目录保留 PR #2528 的运行、审计脚本和 CI 测试依赖。100-case 的唯一维护清单已经上移到
[`cases/ci/manifest.json`](../../../../../manifest.json)，通用 benchmark 驱动位于
[`costmodel_tests/benchmarks/`](../../../../../../../costmodel_tests/benchmarks/)。

完整说明：[CostModel 测试用例与脚本使用说明](../../../../../../../costmodel_tests/README.md)。

在仓库根目录执行：

```bash
python costmodel_tests/run.py list
python costmodel_tests/run.py --case softmax_original --device 0
python costmodel_tests/run.py --device 0
python costmodel_tests/run.py --device 0 --jobs 2 --retries 3
python costmodel_tests/run.py --manifest cases/ci/manifest_all.json --device 0
```

`manifest_all.json` 是尚未验证的 CI 全量清单，只有显式指定才运行；默认使用 CI 根目录的 100-case 清单。

请先加载兼容 PR 的 NPU 编译环境。默认只检查 Auto 最终选择路线的计算精度和 ±10%
耗时预测误差，与原 PR 的 100-case 结果表范围一致。加 `--validation-scope all` 可扩展到
完整路线验收。没有直接断言 Auto 必须是实测最快路线。
