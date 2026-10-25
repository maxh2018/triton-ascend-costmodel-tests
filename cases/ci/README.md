# CI 用例

| 清单 | 内容 | 默认运行 |
|---|---|---|
| [manifest.json](manifest.json) | 原始 PR 100 个 case | 是 |
| [manifest_all.json](manifest_all.json) | 100 个配置 + 339 个 pytest 文件 | 否，Auto/耗时尚待验证 |
| [manifest_excluded.json](manifest_excluded.json) | 暂时排除的 36 个失败文件，含 7 个精度、26 个其他功能、3 个收集失败 | 否，仅供补验 |

`manifest.json` 是 100 个泛化 case 的唯一维护清单。清单只定义 case、shape、参数和
train/holdout 标签；复用的 benchmark 驱动统一放在
[`costmodel_tests/benchmarks/`](../../costmodel_tests/benchmarks/)；仅针对某个
`StageModelKind` 的定制实验放在 `cases/personal/`，不能直接混入泛化门禁。

在仓库根目录执行：

```bash
# 默认：只运行原始 100 个 case
python costmodel_tests/run.py --device 0

# 5 卡快速运行默认 100 case；任一卡忙时直接拒绝启动
python costmodel_tests/run.py --jobs 5 --devices 0,1,2,3,4 --retries 0 \
  --require-idle-devices --timing-source auto

# 查看全量清单；运行时去掉 list
python costmodel_tests/run.py list --manifest cases/ci/manifest_all.json
```

完整目录层级保留用于测试依赖。深层目录中的历史矩阵仅作兼容快照，后续增删 case
只修改本目录的 `manifest.json`。
全量清单通过显式 `--manifest` 选择，不会自动启用。更新清单：`python costmodel_tests/update_ci_manifests.py`。

`manifest_excluded.json` 记录 2026-10-21 功能检查失败的文件；按文件整体排除，源码保留。
其中 `exclusion_reason` 是排除原因，`failed_tests` 是当时失败的测试项。
修复后从这份清单的 `cases` 中删除对应条目，再更新清单，即可恢复。

个人算子放到 `cases/personal/<姓名>/`。

[使用说明](../../costmodel_tests/README.md)
