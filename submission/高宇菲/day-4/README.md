1. 合成数据

合成之后的数据：data/collected_data.jsonl

使用指令：src/data_collect.py 87行

2. 训练

训练代码：src/train.py

3. 环境

/remote-home1/yfgao/miniconda3/envs/py312

合成数据时使用了3张GPU，其中一张单独用来模拟搜索引擎。

训练时用了两张GPU。

4. 权重路径

/remote-home1/yfgao/SummerQuest-2025/submission/高宇菲/day-4/src/qwen25-lora-final

5. 方案

合成如下格式数据：
```json
{
    "prompt": "当前世界上最大的数据中心在哪里？",
    "output": 
    <think> xxx(r1 思考) </think> 
    <tool_call>xxx(格式正确的r1工具调用)</tool_call>
    <think> xxx(r1 得到工具调用结果后的第二次思考) </think> 
    xxx(正式回答)
}
```
其中模拟搜索工具在src/search.py，仍使用r1-7b。提示词在46行。模拟工具使用一张GPU单独部署。

训练时以prompt字段为输入，output字段为目标输出。20%测试集。
