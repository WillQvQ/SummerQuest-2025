# 吴奇峰第五天作业小结

完成了作业一和作业二。

尝试了作业四，完成了vllm合成数据，llamafactory训练模型，以及opencompass评测的流程。**更多是熟悉llamafactory和opencompass的使用，效果不好**，所以代码没有提交。

## 合成数据
从网上下载了三百个来自百度贴吧的问题，使用Qwen3-32B，在思考模式下合成答案

## 训练模型（SFT）
训练的基模型为Qwen2.5-0.B-Instruct，使用llamafactory训练

- LoRA微调，并通过llamafactory的合并功能，对adapter和模型进行了合并
- 全量微调

## 评测模型
探索了opencompass的使用，简单对原模型，LoRA微调得到的模型，全量微调得到的模型，在gsm8k数据集上做了评测。

因为训练集和数学完全无关，所以效果不好