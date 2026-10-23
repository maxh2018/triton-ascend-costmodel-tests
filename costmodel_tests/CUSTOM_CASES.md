# 自定义测试用例提交规范

**自定义测试用例提交流程可概括为先准备一个 Python 测试脚本，然后在 JSON 清单中登记，再按清单运行。**

[运行示例](#1-先运行一个示例) · [上传位置](#2-文件放在哪里) · [清单填写](#3-清单怎么写) · [脚本接口](#4-脚本需要满足什么接口) · [运行与验收](#5-运行与查看结果)

## 1. 示例

仓库提供一个向量加法 `x + y` 示例：

- [示例脚本](../cases/personal/examples/bench_custom_add_example.py)
- [示例清单：1 个 case](../cases/personal/examples/custom_manifest.example.json)

示例清单的字段含义：

| 字段 | 示例值 | 含义 |
|---|---|---|
| `cases` | `[{...}]` | 用例列表；每个对象代表一个测试配置 |
| `name` | `custom_add_f32_n1024` | 清单内唯一的用例名，`--case` 填这个值 |
| `runner`（可选） | `benchmark` | 告诉运行器如何执行脚本；省略时为 `benchmark`，已有 pytest 文件可选 `pytest`，区别见第 4 节 |
| `driver` | `bench_custom_add_example.py` | Python 脚本路径，相对于清单所在目录 |
| `arguments` | `["--n-elements", "1024"]` | 脚本参数，按顺序填写，所有元素均为字符串；这里表示 1024 个元素 |
| `family` | `elementwise` | 算子类别，用于分类统计；这里表示逐元素计算 |
| `split` | `holdout` | 可选，默认 `holdout`（留出验证）；`train` 表示模型开发用例，仅作为标签 |

加载兼容的 NPU 编译环境后，在**仓库根目录**执行：

```bash
python costmodel_tests/run.py --manifest cases/personal/examples/custom_manifest.example.json \
  --case custom_add_f32_n1024 --device 0
```

执行流程：**生成输入 → 运行 kernel → 检查精度 → 采集耗时 → 输出结果**。

添加自己的算子时，可以复制这两个文件，替换 kernel、参考结果和清单中的名字、参数。

## 2. 文件放在哪里

**个人算子放到 `cases/personal/<你的名字>/`，在 `cases/personal/manifest.json` 登记。**

```text
cases/personal/
  manifest.json                  # 个人用例清单，多人共用
  zhangsan/
    bench_custom_my_add.py        # kernel 和测试脚本
```

| 来源 | 存放目录 |
|---|---|
| 个人用例 | `cases/personal/<你的名字>/` |
| 算子仓用例，如 vLLM、SGLang | `cases/operator_repos/` |
| 已有 CI 用例 | `cases/ci/`；个人用例不要放到这里 |

提交时注意：

- **命名**：建议用字母、数字和下划线，清单中的用例名不能重复。
- **来源与依赖**：在脚本头部注明来源、额外依赖和适用芯片；输入尽量在脚本中生成。
- **辅助文件**：随脚本一起提交，按脚本所在位置读取；不要提交编译缓存、profiler 数据和日志。
- **复用计时工具**：通过 `costmodel_tests/run.py` 启动测试时，个人脚本可以直接导入 `from profile_utils import profile_kernel`，使用仓库已有的计时函数，无需复制工具文件。

## 3. 清单怎么写

在 `cases/personal/manifest.json` 的 `cases` 列表中追加个人用例，**保留清单中已有的条目**：

```json
{
  "cases": [
    {
      "name": "custom_my_add_f32_n1024",
      "runner": "benchmark",
      "driver": "zhangsan/bench_custom_my_add.py",
      "arguments": ["--n-elements", "1024"],
      "family": "elementwise",
      "split": "holdout"
    }
  ]
}
```

填写时记住三点：

- **路径**：`driver` 相对于清单所在目录；提交仓库时使用相对路径。本机临时清单也支持绝对路径。
- **参数**：`arguments` 只填脚本自定义参数。benchmark 的 `route`、`factor`、`report`、`profile-dir`、`warmup`、`repeats` 由运行器追加。
- **配置**：同一 kernel 的不同 shape 可以分别登记，使用不同的 `name` 和 `arguments`。

同一清单可以用 `../` 引用不同目录的用例，参考 [混合清单示例](mixed_manifest.example.json)。显式 `--manifest` 优先于 `--category`。

## 4. 脚本需要满足什么接口

`runner` 选择脚本的执行方式。两种方式都检查计算精度和耗时预测误差，按脚本写法选一种即可。

| 对比项 | `benchmark`（默认） | `pytest`（可选） |
|---|---|---|
| 适合谁 | 按本仓库模板编写独立测试脚本的用户 | 已有 `test_*.py`、`test_*()` 测试的用户 |
| 执行方式 | `python bench_custom_my_add.py ...` | `python -m pytest test_add.py ...` |
| 脚本要做什么 | 接收公共参数，检查精度、采集耗时并输出约定 JSON | 保留 pytest 测试函数和精度断言，由适配器采集耗时及生成报告 |
| 清单怎么填 | `"runner": "benchmark"`，也可省略 | `"runner": "pytest"` |

新写用例可以从 benchmark 示例开始；已有 pytest 文件时，可直接看下面的可选接入说明。

### 新写 benchmark 脚本

`benchmark` 表示独立脚本接口，精度检查和计时都由脚本完成。

建议复制 [向量加法示例](../cases/personal/examples/bench_custom_add_example.py) 或 [现有 benchmark 模板](../cases/ci/third_party/ascend/unittest/costmodel_ut/costmodel_generalization/bench_muls_add_costmodel.py)，保留参数解析、选路、精度检查、计时和 JSON 输出部分。

**一个 benchmark case 对应一个被测 kernel。** 脚本需接受以下公共参数；自己的 shape、dtype 等参数可以另外添加。

| 参数 | 要求 |
|---|---|
| `--route` | 支持 `auto`、`simd`、`simt_only` |
| `--factor` | 支持 `1`、`2`、`4`，传给 kernel 的 `superblock_factor` |
| `--warmup`、`--repeats` | 预热和重复计时次数 |
| `--report` | Auto 模式下编译器选路报告的输出路径 |
| `--profile-dir` | 可选；提供时采集 NPU profiler，不提供时也应能运行 |

#### 编译与选路

- 按 `--route` 和 `--factor` 编译运行，不要固定一种路线。
- Auto 设置 `compile_mode="simd_simt"`、`auto_simt_scope_mode="auto"`，将 `--report` 传给 `auto_simt_scope_dump`；显式路线关闭自动选路。
- `logical_program_count_hint` 填真实逻辑 program 数，物理核数从设备属性获取。
- 选路报告由兼容编译器生成，不要自行填写预测耗时。暂不要求增加 Mixed 等额外模式。

#### 精度与计时

- 使用参考结果检查输出，例如 `torch.testing.assert_close()`；失败时抛异常或返回非零退出码。
- 各路线保持相同计算语义，不能跳过不支持的路线后报告成功。
- 提供 `--profile-dir` 时，使用 `profile_kernel(launch, args.profile_dir, kernel_name)` 采集实测耗时。

#### 结果输出

最后向标准输出打印一个 JSON 对象，至少包含：

```python
print(json.dumps({
    "case": "custom_my_add_f32_n1024",
    "correct": True,        # 仅在真实精度检查通过后填写 True
    "profiler": profiler,   # profile_kernel() 返回值；未采集时为 None
}))
```

`profiler` 包含实测耗时 `average_us`，单位为微秒。可以在此前打印日志，最终结果后不要再打印其他 JSON。

### 可选：接入已有 pytest 文件

**仅在你已经有 pytest 测试时使用；采用 benchmark 写法的用户可跳过本节。**

这种方式让已有测试直接加入清单，不必改写为 benchmark 脚本。保留原有 pytest 写法和精度断言，将条目的 `runner` 设为 `pytest`：

```json
{
  "name": "zhangsan_add",
  "runner": "pytest",
  "driver": "zhangsan/test_add.py",
  "arguments": []
}
```

pytest 适配器保留原断言，并验收 Auto 耗时预测。用例需实际运行 kernel 并检查精度；直接调用编译后 kernel 等特殊形式可能需要专门适配。`UNSUPPORTED` 不算通过。

**可选：只检查功能。** 将条目的 `arguments` 设为 `["--costmodel-functional-only"]`，按原 pytest 调用和断言执行，不注入 Auto 选路、不采集或验收耗时。报告中的 Timing 显示 `Not checked`；省略此选项时仍按默认标准验收。


