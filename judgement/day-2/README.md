# Day-2 作业检查脚本

## 功能说明

`llm_lark_checker.py` 是用于检查 Day-2 作业的自动化脚本，使用 DeepSeek 大模型来统计学生收集的论文数量。

### 检查项目

1. **论文收集检查** (Day-2-raw: 论文数量)
   - 扫描学生 `day-2` 目录下的所有 Markdown 文件
   - 合并所有文件内容
   - 使用 DeepSeek 大模型智能统计论文数量
   - 支持多种论文格式识别（标题、ArXiv链接、摘要等）

### 评分规则

- **Day-2-raw**: 根据实际收集的论文数量进行评分
- 模型会智能识别论文标题、ArXiv链接、摘要等内容
- 自动去重和验证论文的有效性

## 使用方法

### 环境准备

1. 设置飞书应用环境变量：
```bash
export FEISHU_APP_ID="your_app_id"
export FEISHU_APP_SECRET="your_app_secret"
```

2. 设置 DeepSeek API 密钥：
```bash
export DEEPSEEK_API_KEY="your_deepseek_api_key"
```

3. 确保学生作业目录结构正确：
```
../../submission/
├── 学生姓名1/
│   └── day-2/
│       ├── paper1.md
│       ├── paper2.md
│       └── ...
├── 学生姓名2/
│   └── day-2/
│       ├── collected_papers.md
│       └── ...
└── ...
```

### 运行脚本

```bash
cd /path/to/judgement
python3 day-2/llm_lark_checker.py
```

### 测试功能

如需测试 DeepSeek API 调用，可以修改脚本末尾：
```python
if __name__ == '__main__':
    test_deepseek_api()  # 启用测试
    # main()             # 禁用正式运行
```

## 功能特性

- **智能识别**: 使用大模型智能识别论文内容，支持多种格式
- **自动合并**: 自动合并学生目录下所有 Markdown 文件
- **避免重复**: 如果飞书表格中已有该学生的数据，跳过重复判断
- **批量更新**: 使用飞书 API 批量更新表格数据
- **详细日志**: 提供详细的检查过程和文件读取状态
- **错误处理**: 完善的异常处理和错误提示

## 输出示例

```
🔍 开始检查学生 张三 的Day-2作业...
📁 找到 3 个Markdown文件:
   1. paper_collection.md
   2. arxiv_papers.md
   3. survey_papers.md

📄 读取文件: paper_collection.md (大小: 15420 bytes)
✅ 成功读取 paper_collection.md

📄 读取文件: arxiv_papers.md (大小: 8930 bytes)
✅ 成功读取 arxiv_papers.md

📄 读取文件: survey_papers.md (大小: 12150 bytes)
✅ 成功读取 survey_papers.md

🤖 使用DeepSeek模型统计论文数量...
📊 模型分析结果: 发现 15 篇有效论文
✅ 统计结果: 15 篇论文

📊 学生 张三 收集的论文总数: 15
📈 统计完成，该学生收集了 15 篇论文

📈 更新统计:
   - 成功更新: 25 人
   - 跳过更新: 3 人
   - 表格中不存在: 1 人
```

## 注意事项

1. 确保飞书应用有足够的权限访问多维表格
2. 确保 DeepSeek API 密钥有效且有足够的调用额度
3. 网络连接稳定，避免 API 调用失败
4. 学生姓名必须与飞书表格中的姓名完全一致
5. 建议在测试环境中先验证 API 调用的正确性
6. 大模型统计结果可能存在误差，建议人工抽查验证

## 技术特点

- **大模型驱动**: 使用 DeepSeek 模型进行智能内容分析
- **多格式支持**: 支持各种论文引用格式和描述方式
- **容错性强**: 对文件编码、格式等问题有良好的容错处理
- **性能优化**: 批量处理和缓存机制提高处理效率