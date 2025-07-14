### Day-5 HW 5-4

#### Steps

- **Qwen2.5-Math-7B** use **GSM8K** to generate **COT** answers for later training.
- Use **LlaMA_Factory** to train **Qwen2.5-0.5B**.
- Utilize **OpenCompass** to evaluate its **long thoughts ability**.

#### Generating Dataset

- Key Codes

```python

PROMPT_TEMPLATE = (
    "You are a math expert. Please solve the question step by step and put the final answer in \\boxed{{}}.\n\nQuestion: {question}"
)

# ...

responses = generate_batch_responses(prompts, batch_size)

# 保存结果
for i, (question, response) in enumerate(zip(questions, responses)):
    result = {
        "instruction": "You are a math expert. Please solve the question step by step.",
        "input": question,
        "output": response
    }
    f.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(f"✅ Saved response for question {i+1}/{len(questions)}")

```

#### LF Training

- llamafactory.sh

```bash

#!/bin/bash
# train_sft.sh
cd LLaMA-Factory

set -x

MODEL_PATH=/data-mnt/data/chwang/models/Qwen2.5-0.5B

llamafactory-cli train \
    --model_name_or_path ${MODEL_PATH} \
    --trust_remote_code \
    --stage sft \
    --do_train \
    --finetuning_type lora \
    --lora_rank 8 \
    --lora_target all \
    --dataset gsm8k_cot \
    --template llama3 \
    --cutoff_len 2048 \
    --max_samples 1000 \
    --overwrite_cache \
    --preprocessing_num_workers 16 \
    --dataloader_num_workers 4 \
    --output_dir saves/qwen2.5-0.5b/lora/sft \
    --logging_steps 10 \
    --save_steps 500 \
    --plot_loss \
    --overwrite_output_dir \
    --save_only_model false \
    --report_to none \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-4 \
    --num_train_epochs 3.0 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --bf16 \
    --ddp_timeout 180000000

```

#### OpenCompass Evaluation

**Notice:** STILL DEBUGGING

- Evaluation Settings (*opencompass_eval.py*)

```python

# eval_config.py

from opencompass.models import HuggingFaceCausalLM, VLLM
from opencompass.datasets import GSM8KDataset

from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate

# Reader 只需指定输入列和输出列
gsm8k_reader = dict(
    input_columns=['question'],
    output_column='answer'
)

models = [
    # 教师模型（vLLM 后端，多卡）
    dict(
        type=VLLM,
        abbr='qwen2.5-math-7b',
        path="/data-mnt/data/chwang/models/Qwen2.5-Math-7B",
        max_out_len=1024,
        max_seq_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=2)
    ),
    # 原始小模型 (Transformer 后端，单卡)
    dict(
        type=HuggingFaceCausalLM,
        abbr='qwen2.5-0.5b-raw',
        path="/data-mnt/data/chwang/models/Qwen2.5-0.5B",
        max_out_len=1024,
        batch_size=16,
        run_cfg=dict(num_gpus=1)
    ),
    # SFT 后小模型 (Transformer 后端，单卡)
    dict(
        type=HuggingFaceCausalLM,
        abbr='qwen2.5-0.5b-sft',
        path="LLaMA-Factory/saves/qwen2.5-0.5b/lora/sft",
        max_out_len=1024,
        batch_size=16,
        run_cfg=dict(num_gpus=1)
    )
]

datasets = [
  dict(
    type=GSM8KDataset,
    abbr='gsm8k',
    path='opencompass/gsm8k',
    reader_cfg=gsm8k_reader,
    infer_cfg=dict(
      retriever=dict(type='ZeroRetriever'),
      inferencer=dict(type='GenInferencer'),
      prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[dict(role="HUMAN", prompt="{question}")])
      ),
    ),
    eval_cfg=dict(
      evaluator=dict(type='AccEvaluator'),
      pred_postprocessor=dict(type='first_capital_postprocess'),
    )
  )
]


```

- *opencompass.sh*

```bash

cd opencompass

opencompass ../opencompass_eval.py

```