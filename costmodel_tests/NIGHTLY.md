# 每晚测试与第二天查看结果

每天只需要做两件事：**晚上执行测试，早上打开看板**。

## 1. 测试同事每晚执行

进入仓库，加载当天要验证的 TA、CANN 和 bishengir 环境，然后执行：

```bash
python costmodel_tests/nightly.py --device 1
```

如果需要关闭 SSH 后继续运行，用后台命令：

```bash
mkdir -p costmodel_test_results
nohup python -u costmodel_tests/nightly.py --device 1 \
  > costmodel_test_results/nightly-launch.log 2>&1 &
```

同一结果目录已有夜测运行时，会拒绝重复启动。

脚本默认运行 `cases/ci/manifest.json` 中的 **原始 100 个 case**，采用固定验收协议：

| 项目 | 设置 |
|---|---|
| 路线 | Auto，使用 TA 实际选中的路线 |
| 实测时间 | Profiler kernel 平均耗时，不会因用例多而切换成 Event |
| 测量次数 | 3 个独立进程，选取中位结果 |
| 预热 / 重复 | 3 / 7 |
| 通过标准 | 计算精度通过，且逐用例 APE ≤10% |
| 并发 | 单卡、逐 case 串行；开始前检查卡是否被其他进程占用 |
| 重试 | 默认失败后额外重试 3 次，只重试失败 case；保留每次尝试 |
| 时限 | 整轮最多 8 小时；超时后保存已有结果，标记未完成 |

TA 和 profile 的选择沿用[现有规则](PROFILES.md)：默认使用当前安装 TA 的参数，显式 `--profile` 用于历史基线。
脚本不安装软件、不切换 conda 环境、不重置设备。

### 常用参数

```bash
# 为当晚的被测版本加备注，并把结果保存到固定目录
python costmodel_tests/nightly.py --device 1 \
  --label "TA-当日构建号" --output-root /path/to/nightly

# 首次结果自检，不自动重试
python costmodel_tests/nightly.py --device 1 --retries 0
```

| 参数 | 默认值 | 用途 |
|---|---|---|
| `--device` | 必填 | 本轮独占使用的物理卡号 |
| `--output-root` | `costmodel_test_results/nightly` | 固定保存目录；每天继续使用同一目录，历史不会被覆盖 |
| `--label` | 空 | 当天的构建号或备注 |
| `--retries` | `3` | 失败后的额外重试次数 |
| `--timeout-hours` | `8` | 整轮最长运行时间 |
| `--manifest` | CI 100-case 清单 | 需要验证另一份 benchmark 清单时显式指定 |
| `--profile` | 当前安装 TA 的 profile | 指定历史参数基线 |
| `--expected-bishengir` | PATH 中的编译器 | 指定预期的 `bishengir-compile` 路径 |
| `--triton-root` / `--npuir-root` | 沿用测试入口默认值 | 可填写被测编译器源码目录，用于记录源码状态 |

**退出码：** `0` 表示全部用例完成正式验收并通过；非 `0` 表示未通过或未完成。以看板中的分类判断原因。

## 2. 第二天怎么看

把结果目录复制到电脑，打开其中的：

```text
costmodel_test_results/nightly/index.html
```

页面无需联网、无需安装前端依赖。**复制整个 `nightly` 目录**可以同时查看日志和原始数据链接。

看板提供：

- **最新批次状态**：通过、未通过、环境阻塞、运行中或结果不完整。
- **历史记录**：选择日期 / 批次，查看最近批次通过数的变化。
- **逐用例结果**：精度、预测时间、实测时间、APE、尝试次数及日志。
- **筛选和排序**：按用例名称搜索，只看失败，按 APE 从大到小排列。
- **版本信息**：实际加载的 TA、profile 版本和哈希，以及测试仓提交。

### 在同一页面查看各用例 JSON

点击表格中的 **JSON** 按钮，或在页面下方的 **原始 JSON 集中查看** 中选择用例，即可查看：

- `route.json`：TA 的完整选路和预测报告。
- `evidence.json`：精度、实际计时和独立测量记录。
- 审计结果、单用例汇总、清单和矩阵记录等验收相关 JSON。

默认先显示最后一次尝试；文件下拉框中也会保留早期尝试。可自动换行、下载所选原文件，完整内容不会截断。
较大的 JSON 按用例单独加载，不集中塞入首页；不扫描编译缓存或 profiler trace。

**下载全部用例指标汇总 JSON** 提供本批次全部 case 的精度、时间、APE 和 JSON 文件索引；各原始文件仍保留在结果目录。
没有执行或没有结果的用例会显示“尚无 JSON”，不会生成虚假的测量数据。

### 团队通过链接查看（可选）

把整个 `nightly` 目录放到已有内网静态 Web 服务中，即可共享 `index.html` 链接。
也可以在**保存了结果目录的机器**上临时提供服务：

```bash
python -m http.server 8000 --bind <本机内网IP> --directory /path/to/nightly
```

同事访问 `http://<本机内网IP>:8000/`。若服务运行在 Docker 内，需要已有端口映射，或先将结果目录复制到提供 Web 服务的机器。
本脚本不会自动上传或公开日志。

## 3. 如何解读结论

| 页面结论 | 含义 |
|---|---|
| 通过 | 精度通过，使用有效 Profiler 实测，APE ≤10%，运行与审计均通过 |
| 时间误差超标 | 精度通过，但 APE >10% |
| 精度失败 | 计算结果未满足用例自身的精度要求 |
| 执行异常 | 编译、运行或结果文件存在问题，不直接算作精度失败 |
| 环境阻塞 | 设备启动失败、设备 Critical、profile 预检等问题 |
| 未执行 | 没有完成该用例；不能算作验收通过或精度失败 |
| 未完成时间验收 | 缺少有效预测、Profiler 时间或时间验收记录；Event 数据不会算作正式通过 |

APE 的分母是实测时间：`abs(预测时间 - 实测时间) / 实测时间 × 100%`。
每行展示最后一次尝试；**尝试次数大于 1** 表示发生过重试，原始尝试记录仍然保留。

### 结果文件

```text
nightly/
├── index.html                 # 总看板：最新结果与历史批次
└── runs/<日期_时间>/
    ├── index.html             # 这一晚的独立看板
    ├── dashboard.json         # 看板使用的逐用例汇总
    ├── case_json/             # 各用例的离线 JSON 浏览页
    ├── run.json               # 执行命令、时间、版本及设备检查
    ├── nightly_manifest.json  # 本轮清单快照
    ├── nightly.log            # 完整测试入口日志（已启动测试时生成）
    └── raw/                   # 原始运行、审计、profile 快照和重试记录
```

即使预检失败，也会生成看板，显示失败原因和未执行数量。不会沿用上一晚的通过状态。
在运行中刷新页面，可以查看已完成结果。

只重新生成看板、不运行 NPU 测试：

```bash
python costmodel_tests/dashboard.py --output-root costmodel_test_results/nightly
```

每天开始测试、第二天查看结果按上述流程人工执行；本交付不自动配置定时任务或 Web 服务。
