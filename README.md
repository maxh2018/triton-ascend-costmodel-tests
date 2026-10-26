# Triton-Ascend CostModel Tests

本仓库保存 CostModel 测试用例、测试依赖、profile 和说明文档。
运行测试需要单独安装兼容的 Triton-Ascend、CANN 和 NPU 编译环境。

本仓库是泛化用例的唯一维护源：CI 泛化 case 定义维护在
[`cases/ci/manifest.json`](cases/ci/manifest.json)，通用 benchmark 驱动维护在
[`costmodel_tests/benchmarks/`](costmodel_tests/benchmarks/)，针对单个
`StageModelKind` 的探索性用例维护在 `cases/personal/`。

- [测试脚本使用说明](costmodel_tests/README.md)
- [自定义测试用例提交规范](costmodel_tests/CUSTOM_CASES.md)
- [Profile 的作用与版本管理](costmodel_tests/PROFILES.md)

## 获取与运行

```bash
git clone https://github.com/maxh2018/triton-ascend-costmodel-tests.git
cd triton-ascend-costmodel-tests

# 加载编译环境后，运行默认 CI 100-case 清单
python costmodel_tests/run.py --device 0
```

开发者从 `main` 创建开发分支，按[提交规范](costmodel_tests/CUSTOM_CASES.md)添加用例和清单，
验证后向本仓库的 `main` 提交 PR。

## 用例分类

用例按来源分目录保存：

| 分类 | 路径 | 说明 |
|---|---|---|
| CI 用例 | [cases/ci/](cases/ci/README.md) | 现有 pytest_ut、PR 的 100-case 和测试依赖 |
| 个人用例 | [cases/personal/](cases/personal/README.md) | 个人算子上传到 `<姓名>/` 子目录，登记到此处 `manifest.json` |
| 算子仓用例 | [cases/operator_repos/](cases/operator_repos/README.md) | 按 vLLM、SGLang 等来源仓库分目录保存 |
## 资料自检

无需 NPU 的资料自检：

```bash
python -m unittest discover -s costmodel_tests -p 'test_*.py' -v
```

## 来源

初始文件迁自 [maxh2018/triton-ascend-mxh 的测试分支](https://github.com/maxh2018/triton-ascend-mxh/tree/dab9ba39ab10dca45b7cc4113846dd562b6fc370)。
原始 CI 用例、PR #2528 内容和迁移提交记录在 [sources.json](costmodel_tests/sources.json) 中。
原有 [LICENSE](LICENSE) 和 [版权说明](Open_Source_Software_Notice) 随文件保留。
