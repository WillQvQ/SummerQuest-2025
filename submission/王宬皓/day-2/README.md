> github: https://github.com/henrywch/SummerQuest-2025.git

### Day-2 HW

#### MCP Settings

- *.vscode/mcp.json*

```json
{
    "servers": {
        "my-mcp-server-fad6bb35": {
            "type": "stdio",
            "command": "/root/miniconda/envs/vllm/bin/python",
            "args": ["/root/SummerQuest-2025/submission/王宬皓/day-2/trae_paper_collector/arxiv_mcp_server.py"]
        }
    }
}
```

- *.github/copilot-instructions.md*

```markdown
我会给你一篇或多篇论文的 ArXiv 链接或名字。
1. 如果我给你的是论文名，你需要先使用搜索引擎获得它的 ArXiv 链接。
2. 必须使用 arxiv-citation-analyzer 这个工具来获取至少十篇相关论文的摘要。
3. 你需要根据摘要和我告诉你的指令，来判断这些相关论文是否符合我指令中的描述。
4. 你需要最终把符合这些描述的论文整理出来，生成 markdown 格式的文件，存到 ./md_files/ 目录下。

文件中的内容格式参考以下示例：

## 需求1-相关

### RAGEN: Understanding Self-Evolution in LLM Agents via Multi-Turn Reinforcement Learning
- ArXiv链接 : https://arxiv.org/abs/2504.20073
- 关键特点 : 提出了StarPO框架，用于轨迹级代理RL，并引入RAGEN系统用于训练和评估LLM代理。
- 相关技术 : Multi-Turn Reinforcement Learning, StarPO

### DAPO: An Open-Source LLM Reinforcement Learning System at Scale
- ArXiv链接 : https://arxiv.org/abs/2503.14476
- 关键特点 : 提出了DAPO算法，并开源了一个大规模RL系统，用于LLM的强化学习。
- 相关技术 : DAPO, Large-Scale RL System 

## 需求2-相关

## 需求1 & 需求2 都相关的论文

```

#### Topic: Multimodal Robotics

Results see *2505.21457_multimodal_hri.md*