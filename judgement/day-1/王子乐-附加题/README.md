**Day‑1 额外题——作业自动批改脚本**

脚本功能如下：

1. **文件完整性检查**
    自动遍历每位学生提交的目录，验证 `README.md`、`doc_viewer.py`、各项日志文件等必需文件是否齐全，并记录缺失项。
2. **AI 评分**
    由于本地 vLLM/Qwen 7B 无法胜任此评分任务，故使用我自己购买的 DeepSeek Chat API，结合预设的评分标准（文件完整性、README 质量、代码质量、日志正确性、环境配置），自动生成分数与详细反馈。
3. **并行处理**
    支持多线程并行批改，并自动节流，避免 API 超额。
4. **参数配置**
   - `--submission-dir`：指定学生作业总目录
   - `--api-key` / `--model`：配置 AI 接口
   - `--output-dir`：自定义结果保存路径
   - `--sequential`：取消并行切换为顺序处理
5. **结果输出**
    最终在同一目录下的grading_results中生成：
   - JSON 格式的评分数据（含详情、反馈）
   - Markdown 格式的批改报告（含统计与逐人点评）

运行脚本即可一键完成从文件检查、AI 评分到报告生成的全流程。

**使用方法：**

```
python judge.py #使用默认参数
```

```
python judge.py --submission-dir [SUBMISSION-DIR] --api-key [API-KEY] --model [deepseek-chat] --output-dir [grading_results] --sequential [store_true]
```

