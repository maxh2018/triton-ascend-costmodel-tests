# 算子仓用例

本目录保存来自 vLLM、SGLang 等算子仓的用例。按来源仓库分子目录，
在本目录的 `manifest.json` 中登记。
`driver` 路径相对于本清单；与 CI、个人用例分开保存。

```text
cases/operator_repos/
  manifest.json
  vllm/
    <算子名>/
      bench_<算子名>.py
  sglang/
    <算子名>/
      bench_<算子名>.py
```

```bash
python costmodel_tests/run.py list --category operator_repos
python costmodel_tests/run.py --category operator_repos --device 0
```

当前清单为空，登记用例后才能运行。每个脚本头部注明原仓库、原文件路径和版本。
例如脚本放在 `vllm/rms_norm/bench_rms_norm.py`，清单的 `driver` 就填写这个相对路径。
用例名建议加来源前缀，例如 `vllm_rms_norm_f16_n4096`。
接口要求见 [提交规范](../../costmodel_tests/CUSTOM_CASES.md)。
