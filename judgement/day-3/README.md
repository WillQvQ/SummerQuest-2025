# Day-3 作业检查脚本

## 功能说明

`rule_lark_checker.py` 是用于检查 Day-3 作业的自动化脚本，包含以下功能：

### 检查项目

1. **hw3_1.json 检查** (Day-3-hw1: 0分或2分)
   - 检查文件格式是否正确
   - 验证 `special_tokens` 和 `tasks` 字段是否存在
   - 确认每个 task 的 `token_ids` 中包含至少一个 `special_tokens` 的 id

2. **hw3_2.json 检查** (Day-3-hw2: 0-8分)
   - 检查每个 Output 是否包含 `<think>` 部分
   - 验证是否包含特殊词符 `<|EDIT|>` 或 `<|AGENT|>`
   - 检查函数调用是否正确：
     - `<|AGENT|>` 后应调用 `python` 函数
     - `<|EDIT|>` 后应调用 `editor` 函数

### 评分规则

- **Day-3-hw1**: 0分或2分（hw3_1.json 格式正确且所有 token_ids 包含 special_token_id）
- **Day-3-hw2**: 0-8分（hw3_2.json 中每个正确的 Output 得1分，最多8分）

## 使用方法

### 环境准备

1. 设置飞书应用环境变量：
```bash
export FEISHU_APP_ID="your_app_id"
export FEISHU_APP_SECRET="your_app_secret"
```

2. 确保学生作业目录结构正确：
```
../../submission/
├── 学生姓名1/
│   └── day-3/
│       ├── hw3_1.json
│       └── hw3_2.json
├── 学生姓名2/
│   └── day-3/
│       ├── hw3_1.json
│       └── hw3_2.json
└── ...
```

### 运行脚本

```bash
cd /path/to/judgement
python3 day-3/rule_lark_checker.py
```

### 测试功能

如需测试检查逻辑，可以修改脚本末尾：
```python
if __name__ == '__main__':
    test_hw3_checkers()  # 启用测试
    # main()             # 禁用正式运行
```

## 功能特性

- **智能跳过**: 如果学生没有 `day-3` 目录，自动跳过
- **避免重复**: 如果飞书表格中已有该学生的数据，跳过重复判断
- **批量更新**: 使用飞书 API 批量更新表格数据
- **详细日志**: 提供详细的检查过程和结果统计

## 输出示例

```
🔍 开始检查学生 张三 的Day-3作业...

📄 检查 hw3_1.json...
✅ hw3_1.json 检查通过，得分: 2/2

📄 检查 hw3_2.json...
📊 hw3_2.json 得分: 7/8
✅ hw3_2.json 检查完成，共 10 个项目，通过 7 个，得分 7/8

📈 学生 张三 总分: Day-3-hw1=2/2, Day-3-hw2=7/8

📈 更新统计:
   - 成功更新: 25 人
   - 跳过更新: 3 人
   - 表格中不存在: 1 人
```

## 注意事项

1. 确保飞书应用有足够的权限访问多维表格
2. 网络连接稳定，避免 API 调用失败
3. 学生姓名必须与飞书表格中的姓名完全一致
4. 建议在测试环境中先验证检查逻辑的正确性