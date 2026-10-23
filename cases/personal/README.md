# 个人用例上传位置

**个人算子提交到 `cases/personal/<你的名字>/`，并登记到本目录的 `manifest.json`。**

例如：

```text
cases/personal/
  manifest.json
  zhangsan/
    bench_my_add.py
```

清单中的 `driver` 填 `zhangsan/bench_my_add.py`，相对于清单所在目录。
用例名建议加姓名前缀，例如 `zhangsan_add_f32_n1024`。
脚本接口和 JSON 字段说明见 [提交规范](../../costmodel_tests/CUSTOM_CASES.md)。

在仓库根目录运行：

```bash
python costmodel_tests/run.py list --category personal
python costmodel_tests/run.py --category personal --case zhangsan_add_f32_n1024 --device 0
python costmodel_tests/run.py --category personal --device 0
```

`manifest.json` 初始为空，登记用例后才能运行。
`examples/` 是可复制的向量加法示例，默认不加入个人用例清单；运行示例时显式指定：

```bash
python costmodel_tests/run.py --manifest cases/personal/examples/custom_manifest.example.json \
  --case custom_add_f32_n1024 --device 0
```
