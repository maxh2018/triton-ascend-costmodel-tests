# CostModel 测试使用说明

[通用测试方法](#1-通用测试方法) · [快速验证命令](#2-快速验证命令) · [结果与验收](#3-结果与验收)

## 1. 通用测试方法

### 按清单运行

先用 JSON 清单列出要测的脚本和输入参数，再把清单交给运行器：

```bash
python costmodel_tests/run.py --manifest my_tests.json --device 0
```

运行器按清单执行用例，检查**计算精度**和**耗时预测误差**，最后生成结果报告。

以下命令均在**仓库根目录**执行。实际测试需要已加载兼容的 NPU 编译环境；
`list` 和 `--dry-run` 不需要 NPU。

### 清单结构与字段

例如，将下面内容保存为仓库根目录的 `my_tests.json`，即可运行已有的向量加法示例：

```json
{
  "cases": [
    {
      "name": "custom_add_f32_n1024",
      "runner": "benchmark",
      "driver": "cases/personal/examples/bench_custom_add_example.py",
      "arguments": ["--n-elements", "1024"],
      "family": "elementwise",
      "split": "holdout"
    }
  ]
}
```

| 字段 | 必填 / 默认值 | 含义 |
|---|---|---|
| `cases` | 必填 | 用例列表；每个对象是一个运行配置，追加对象即可增加用例 |
| `name` | 必填 | 清单内唯一的名字；`--case` 按这个名字筛选，建议使用字母、数字和下划线 |
| `runner` | `benchmark` | 脚本接口：`benchmark` 使用仓库 benchmark 接口；`pytest` 运行 pytest 文件并接入计时适配器 |
| `driver` | 必填 | Python 脚本路径，**相对于清单所在目录**，也支持 `../` 和绝对路径 |
| `arguments` | `[]` | 脚本参数列表，所有元素写成字符串；例如 `"--n-elements", "1024"` |
| `family` | `custom` | 算子类别，用于分类统计；如 `elementwise`、`reduction` |
| `split` | `holdout` | `train` 为模型开发用例，`holdout` 为留出验证用例；作为标签记录 |

同一个脚本可以登记多个 shape 配置，分别设置不同的 `name` 和 `arguments`。

可选顶层字段：`description` 是清单说明；`validation_status: "unverified"` 标记尚未验证，不代表测试已通过。

### 命令语法与参数

```text
python costmodel_tests/run.py [run | list] [选项]
```

`run` 执行清单，可以省略；`list` 只列出用例。`-h` / `--help` 查看完整帮助。

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--manifest <JSON路径>` | 无 | **最高优先级**，按指定清单执行；相对路径以当前目录为准，也支持绝对路径 |
| `--category <分类>` | `ci` | 仅在未指定清单时，读取 `cases/<分类>/manifest.json`；支持 `ci`、`personal`、`operator_repos` |
| `--case <名称>` | 整份清单 | 只选指定条目，可重复填写；填 `name`，不填 `.py` 文件名 |
| `--device <卡号>` | 继承环境 | 使用的卡号，例如 `--device 0`；并发用例共用这张卡 |
| `--jobs <数量>` | `1` | 同时执行的清单条目数 |
| `--retries <次数>` | `3` | 失败后的**额外**重试次数；默认最多尝试 4 次，`0` 关闭重试 |
| `--output-root <目录>` | 自动生成 | 保存结果的位置，必须是新目录或空目录 |
| `--dry-run` | 关闭 | 只打印执行计划，不运行测试、不生成结果目录 |

指定 `--manifest` 后忽略 `--category`；清单出错时直接报错，不会回退。

<details>
<summary>环境准备与进阶参数</summary>

使用 Ascend NPU Linux 服务器，加载 CANN、Triton-Ascend、torch_npu 和 bishengir 环境。
本仓库只保存测试资料，编译器需单独安装并支持 PR #2528 的选路参数和报告格式。
Python 基础依赖：`pytest`、`pytest-xdist`、`torch`、`torch_npu`、`triton`，部分用例另有依赖。

```bash
unset TRITON_ASCEND_COMPILE_MODE
```

| 参数 | 含义 |
|---|---|
| `--profile <JSON路径>` | 指定编译和审计共用的 profile；默认读取环境变量 `TRITON_ASCEND_AUTO_SIMT_PROFILE`，否则用仓库的 950PR/A5 profile |
| `--expected-bishengir <文件路径>` | 指定预期的 `bishengir-compile`；默认从 PATH 查找 |
| `--triton-root <目录>` | 记录被测 Triton-Ascend 源码状态，不负责安装或切换 Python 包 |
| `--npuir-root <目录>` | 记录被测 NPU-IR 源码状态，不切换工具链 |
| `--validation-scope auto / all` | 默认 `auto`；`all` 为 benchmark 增加规定的显式路线，pytest 仍验收 Auto |

</details>

## 2. 快速验证命令

下面每组提供**单卡串行**和**单卡并发、失败重试**两种选择。卡号按服务器实际情况修改。
`--jobs 2` 表示最多同时执行两个用例，共用 `--device` 指定的卡；并发可能影响实测耗时，严格验收耗时误差时使用串行命令。

只重试失败用例，已通过的条目不会重跑；同一 pytest 文件内也只重试失败的参数化节点。
收集阶段失败、尚无节点结果时重试该文件；已有通过结果但无法定位失败节点时，不会全量重放。

### 原始 100 个 case

```bash
# 单卡串行；默认清单为 cases/ci/manifest.json
python costmodel_tests/run.py --device 0

# 可选：单卡并发，失败后额外重试 3 次
python costmodel_tests/run.py --device 0 --jobs 2 --retries 3
```

只测一个 case 时追加 `--case softmax_original`；只查看清单用 `python costmodel_tests/run.py list`。

<details>
<summary>可选：CI 全量清单（尚未验证，不默认运行）</summary>

[manifest_all.json](../cases/ci/manifest_all.json) 当前包含 **100 个 benchmark 配置 + 339 个 pytest 文件，共 439 个条目**。
pytest 函数和参数化节点在运行时展开，条目数不代表独立算子数。

2026-10-21 功能检查中失败的 **36 个文件**已暂时移到 [manifest_excluded.json](../cases/ci/manifest_excluded.json)：
7 个精度失败、26 个其他功能失败、3 个收集失败。按文件整体排除，源码保留，生成清单时也会继续排除。

```bash
# 显式选择，单卡执行
python costmodel_tests/run.py --manifest cases/ci/manifest_all.json --device 0

# 可选：单卡并发，失败后额外重试 3 次
python costmodel_tests/run.py --manifest cases/ci/manifest_all.json \
  --device 0 --jobs 2 --retries 3
```

Auto 选路及耗时验收仍待验证；`UNSUPPORTED` 会导致严格验收失败。

修复后，从 `manifest_excluded.json` 的 `cases` 中删除对应条目，再执行
`python costmodel_tests/update_ci_manifests.py`，即可恢复到全量清单。

</details>

### 指定目录的用例

用 `--category` 读取对应目录的默认清单，执行的是**清单中已登记的用例**：

| 分类 | 默认清单 |
|---|---|
| `ci` | `cases/ci/manifest.json` |
| `personal` | `cases/personal/manifest.json` |
| `operator_repos` | `cases/operator_repos/manifest.json` |

个人、算子仓的默认清单初始为空，需先登记用例。以下以个人目录为例：

```bash
# 个人目录：单卡执行
python costmodel_tests/run.py --category personal --device 0

# 可选：个人目录，单卡并发，失败后额外重试 3 次
python costmodel_tests/run.py --category personal --device 0 --jobs 2 --retries 3
```

运行算子仓用例时，将 `personal` 换成 `operator_repos`；个人脚本放在 `cases/personal/<姓名>/`。

### 自定义清单

将第一节的 JSON 保存为 `my_tests.json`，按需在 `cases` 中增加条目，然后执行：

```bash
# 使用自己的清单，单卡执行
python costmodel_tests/run.py --manifest my_tests.json --device 0

# 可选：多个条目时，单卡并发，失败后额外重试 3 次
python costmodel_tests/run.py --manifest my_tests.json --device 0 --jobs 2 --retries 3
```

清单可以跨目录，也可以放在分类目录之外。一个条目时不会因为 `--jobs 2` 而重复运行。
执行前可加 `--dry-run` 查看计划；要关闭重试，把 `--retries` 设为 `0`。

## 3. 结果与验收

### 结果文件

终端打印 `Output: <目录>`，默认结果目录为 `costmodel_test_results/run_<时间>/`。

| 文件 | 看什么 |
|---|---|
| `summary.json` | **整次运行的 PASS/FAIL**、实际清单和退出码 |
| `report.md` | **优先阅读**：用例精度、耗时、尝试次数，以及预测/实测/误差 |
| `report.json` | 完整结果，便于程序统计 |
| `batch_summary.json` | `failed_cases`、每次日志路径、最后一次尝试目录 |
| `batch_summary.partial.json` | 正在执行时的完成进度 |
| `manifest.json`、`costmodel_profile.json` | 本次清单和 profile 快照 |
| `profile_dependencies/` | profile 引用的共享 microbenchmark 数据；归档结果时一起保存 |
| `cases/<名称>/attempt_01.log` | 首次日志，重试依次生成 `attempt_02.log` 等 |

`summary.json` 的 `manifest_source` 为 `explicit`（显式清单）或 `category_default`（默认清单）。
显式清单的 `category` 为 `null`，表示这次运行由清单决定范围。

### 验收标准

1. **精度通过**：运行成功，计算结果满足参考结果和用例自身的精度容差。
2. **耗时通过**：Auto 所选路线的预测与实测耗时齐全，绝对相对误差 ≤ 10%。

```text
relative_error = predicted_us / measured_us - 1
通过条件：abs(relative_error) <= 0.10
```

例如实测 `10 μs`、预测 `10.8 μs`，误差 `+8%`，通过；预测 `11.5 μs` 则为 `+15%`，失败。
**10% 是耗时预测误差，不是精度误差。** 当前不要求 Auto 是实测最快路线。

所有所选条目通过后，`summary.json` 的 `status` 为 `PASS`，退出码为 `0`。
一个用例有多个被测 kernel 配置时，所有采集配置都要通过。
没有实际运行 kernel、缺少预测或计时不能算通过；`UNSUPPORTED` 会使严格验收失败，`SKIP` 不计入通过数。

### 失败排查

1. 打开 `report.md`，确认失败用例及精度、耗时状态。
2. 在 `batch_summary.json` 查 `failed_cases`，按 `attempts[].log` 查看日志。
3. 对照原始报告中的 `predicted_us`、`measured_us`、`relative_error`、`failure_reasons` 定位问题。

需要缩小范围时，可使用 [3-case 调试清单](../cases/ci/manifest.debug.json)，不改动默认 100-case 清单：

```bash
python costmodel_tests/run.py --manifest cases/ci/manifest.debug.json --device 1 --jobs 1 --retries 0
```

若出现 `failed to read microbenchmark profile`，先检查 profile 引用的数据文件。运行器生成快照时会将该文件复制到 `profile_dependencies/` 并更新引用，避免因移动 profile 导致路径失效。

<details>
<summary>原始证据在哪里</summary>

每次尝试位于 `cases/<名称>/attempt_XX/`：

- benchmark：`runner.log`、`audit.log`、`generalization_audit.json`，以及各路线的 `route.json`、`evidence.json`。
- pytest：`pytest.xml` 和 `pytest_costmodel/` 下的报告、逐节点 `result.json`、选路报告和 profiler 数据。

pytest 跨重试合并后的节点结果保存为 `cases/<名称>/pytest_summary.json`。
精度断言和异常堆栈查看运行日志或 `pytest.xml`；`final_output_root` 指向最后一次尝试。

</details>

---

上传规范见 [CUSTOM_CASES.md](CUSTOM_CASES.md)，来源见 [sources.json](sources.json)。
已完成 CPU 回归和 3-case 调试清单的 NPU 实测；完整 100-case 和 CI 全量清单尚未完成本轮验收。示例不表示其他 shape 已通过误差验收。
新增 CI 文件后，可用 `python costmodel_tests/update_ci_manifests.py` 更新清单；该操作不进行测试验证。
