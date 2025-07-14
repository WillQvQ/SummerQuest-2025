# Day 4 作业
## 选择了hw4_3，文件说明如下：
- `day-4.md`：本文件，包含作业内容和说明。
- `day_synthesis.py`：Python代码文件，用于合成训练需要的数据。
- `day_formatter.py`：Python代码文件，用于格式化合成的数据进行训练。
- `train.py`：Python代码文件，用于训练目标模型。
- `fake_search.py`：Python代码文件，用于提供搜素功能接口，使用的是Qwen2.5-3B-Instruct模型。
- `save_question.py`：Python代码文件，用于保存问题。
- `no_search_required_questions.json`,`search_required_questions.json`：原始问题集。
- `synthetic_data_withsearch.json`,`synthetic_data_nosearch.json`：合成的数据集。
- `qwen_tool_train_eval_mixed.jsonl`,`qwen_tool_train_train_mixed.jsonl`：格式化后的验证集和训练集。
- `./qwen_0.5b_tool_lora_mixed/final_adapter`：训练后的权重。