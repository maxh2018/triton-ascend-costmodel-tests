# Profile 的作用与版本管理

## 测试如何使用 profile

profile 是 TA CostModel 的参数文件。TA 分析 kernel 中的运算、访存和控制流，结合这些参数计算预测时间、选择 Auto 路线。

| 文件 | 提供什么 |
|---|---|
| `simd_simt/*.json` | 路线模型的校准参数、启动开销和部分策略配置 |
| `microbench/*.json` | 硬件吞吐率、延迟等实测参数，由上面的文件引用 |
| `*_schema.json` | TA 支持的参数结构及格式版本，用于运行前校验 |

默认 Auto 测试的流程：

```text
kernel + 输入参数
    ↓
TA 使用配套 profile 编译、选路，输出预测时间（μs）
    ↓
运行编译后的 kernel，检查精度、采集 NPU 实测时间（μs）
    ↓
测试脚本读取预测值和实测值，计算 APE 并生成报告
```

`APE = |预测时间 − 实测时间| / 实测时间`。精度容差和耗时验收阈值由测试代码设置。

**profile 与 profiler 不同：** profile 是预测参数；profiler / device event 是实测计时工具。
显式路线的旧审计仍按 TA 候选周期数结合 profile 换算时间；此次修改统一参数来源和校验，保留该审计逻辑。

## 默认跟随实际安装的 TA

```bash
python costmodel_tests/run.py --device 0
```

读取顺序：

1. 显式 `--profile <JSON路径>`。
2. 显式环境变量 `TRITON_ASCEND_AUTO_SIMT_PROFILE`。
3. 当前 Python 环境实际加载的 TA 所携带、且兼容所选 NPU 的 profile。

默认从实际加载的原生扩展旁的 `triton/_C/ascend/costmodel_profiles`、backend 包的 `costmodel_profiles` 查找；
editable 安装也支持实际 backend 源码旁的 `costmodel/profiles`。
不会自动读取测试仓的历史副本，也不会根据 `--triton-root` 猜测安装位置。
若要自动选择，请先清除以前手动设置的环境变量：

```bash
unset TRITON_ASCEND_AUTO_SIMT_PROFILE
```

TA 未携带 profile/schema、找不到兼容配置或存在多份可用配置时，运行前明确报错。
多份配置可用时用 `--profile` 明确选择；缺少随包资源时需要安装包含这些资源的 TA。

## 显式使用历史参数

```bash
python costmodel_tests/run.py --device 0 \
  --profile cases/ci/third_party/ascend/costmodel/profiles/simd_simt/ascend_950pr_simd_simt_v1.json
```

历史文件也必须通过**当前安装 TA 携带的 schema**及目标硬件检查。
旧格式不兼容时会拒绝运行，不会通过关闭校验放行。
指定旧 profile 是使用历史参数；完整复现旧结果还需要相同的 TA 构建、工具链和硬件。

## 运行前检查什么

- 检查每张指定 NPU 的目标型号是否在 profile 的 `compatible_targets` 中。
- 用已安装 TA 的 schema 校验路线参数和 microbenchmark 参数。
- 检查两份参数的目标一致、引用的 measurement ID 存在。
- 固定通过校验的文件内容；子任务复用本次快照，并检查哈希和 TA 安装状态。

纯功能模式 `--costmodel-functional-only` 和 pytest 仅收集用例时不执行 CostModel，因此不要求 TA/profile 校验，也不会据此判定耗时通过。
`list`、`--dry-run` 仍无需 NPU 环境。

## 从结果查看实际使用的版本

优先查看 `report.md` 的 **Profile and TA** 段落；详细信息在 `profile_metadata.json`：

| 信息 | 用途 |
|---|---|
| `source_kind`、`source_path` | 自动发现还是显式覆盖，以及原始位置 |
| `profile_version`、`schema_version`、`source_sha256` | 原始参数的版本及 SHA256 |
| `runtimes` | 实际加载的 TA/Triton 版本（含可确认归属的 wheel 构建版本）、路径、代码/原生库指纹和每张卡的目标 |
| `artifacts` | 本次完整快照中各文件的 SHA256 |

快照包含运行时读取的全部参数及校验 schema：

```text
<结果目录>/
├── profile_metadata.json
├── costmodel_profile.json                  # 运行使用，依赖路径已改为快照内相对路径
├── profile_sources/selection.json          # 原始 profile 字节不变，便于核对原始哈希
├── profile_dependencies/                   # 硬件参数及其 schema
└── profile_schemas/                        # 已安装 TA 的路线参数 schema
```

快照不复制 microbenchmark 的全部原始采样日志；那些是参数来源证据，不是本次预测运行时读取的输入。
