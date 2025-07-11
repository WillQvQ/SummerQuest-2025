## 暑期集训 Day2 小结

### 作业1

本次作业要求使用任意大模型助手配合 arxiv_mcp_server.py 收集某个小方向的若干论文，整理到 papers.md 中。我实现了 paper_getter.py 遍历了给定源论文的所有一级引用（直接引用和直接被引用），然后使用大模型来阅读该论文的 abstract 给出一个相关性评分，最终选取评分前 10 的文章输出，具体实现上还使用并行处理加速运行。我以源论文《Efficient Memory Management for Large Language Model Serving with PagedAttention》和关键字 “KV-cache” 来进行搜索，将结果保存到了 papers.md 中。