from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
import random
import requests
from tqdm import tqdm
import os
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from torch.utils.data import Dataset, DataLoader

# Configuration - FIXED: Use absolute path with correct format
LOCAL_MODEL_PATH = "/data-mnt/data/downloaded_ckpts/DeepSeek-R1-Distill-Qwen-7B"  # Corrected spelling
NUM_GPUS = 2
BATCH_SIZE = 8
MAX_QUESTIONS = 3000

# Load model using transformers with local_files_only
tokenizer = AutoTokenizer.from_pretrained(
    LOCAL_MODEL_PATH,
    trust_remote_code=True,
    local_files_only=True  # Crucial for local models
)
model = AutoModelForCausalLM.from_pretrained(
    LOCAL_MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    local_files_only=True  # Crucial for local models
)
model.eval()

# System prompt template
SYS_PROMPT = """You are an AI assistant capable of complex reasoning. Follow this process:
1. Think deeply in the <think> tag
2. Call search when real-time data/unknown knowledge is needed
3. Search call format: <|SEARCH|>"""

# Load SQuAD dataset questions
def load_squad_questions(max_questions=1000):
    try:
        urls = [
            "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json",
            "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"
        ]
        
        all_questions = []
        for url in urls:
            response = requests.get(url)
            squad_data = response.json()
            
            for article in squad_data["data"]:
                for paragraph in article["paragraphs"]:
                    for qa in paragraph["qas"]:
                        all_questions.append(qa["question"])
        
        random.shuffle(all_questions)
        return all_questions[:max_questions]
        
    except Exception as e:
        print(f"Error loading SQuAD dataset: {str(e)}")
        return [
            "How does quantum entanglement affect error correction in quantum computers?",
            "Impact of the 2024 global chip shortage on automakers' electrification strategies",
            "Application and limitations of Bayes' theorem in medical diagnosis",
            "Important experiments currently being conducted on the International Space Station?",
            "How does relativity explain time dilation in GPS systems?"
        ]

# Diversity parameters
diversity_params = {
    "temperature": [0.5, 0.7, 0.9],
    "top_p": [0.8, 0.9, 0.95],
    "repetition_penalty": [1.0, 1.1, 1.2],
    "max_new_tokens": [256, 384, 512]
}

# Custom dataset class
class QuestionDataset(Dataset):
    def __init__(self, questions):
        self.questions = questions
        
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        return self.questions[idx]

# Batch generation function with performance improvements
def generate_batch(questions):
    batch_data = []
    stage1_prompts = []
    params_list = []
    
    # Prepare batch inputs
    for question in questions:
        params = {
            "temperature": random.choice(diversity_params["temperature"]),
            "top_p": random.choice(diversity_params["top_p"]),
            "repetition_penalty": random.choice(diversity_params["repetition_penalty"]),
            "max_new_tokens": random.choice(diversity_params["max_new_tokens"])
        }
        params_list.append(params)
        stage1_prompts.append(
            f"Question: {question}\n"
            "Please think deeply. Output format: <think>Your thoughts</think>\n"
            "Thinking:"
        )
    
    # Batch tokenization - FIXED: Use pad_to_multiple_of for better performance
    inputs = tokenizer(
        stage1_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
        pad_to_multiple_of=8,  # Better GPU utilization
        return_token_type_ids=False
    ).to(model.device)
    
    # Batch generation with optimized settings
    with torch.no_grad(), torch.cuda.amp.autocast():
        stage1_outputs = model.generate(
            **inputs,
            max_new_tokens=max(p["max_new_tokens"] for p in params_list),
            temperature=random.choice(diversity_params["temperature"]),
            top_p=random.choice(diversity_params["top_p"]),
            repetition_penalty=random.choice(diversity_params["repetition_penalty"]),
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True  # Faster generation
        )
    
    # Process batch outputs
    thoughts_list = []
    for i in range(len(questions)):
        # Only decode generated portion
        start_index = inputs.input_ids[i].shape[0]
        thoughts = tokenizer.decode(
            stage1_outputs[i][start_index:], 
            skip_special_tokens=True
        ).strip()
        
        if not thoughts.startswith("<think>"):
            thoughts = f"<think>{thoughts}"
        if not thoughts.endswith("</think>"):
            thoughts = f"{thoughts}</think>"
        thoughts_list.append(thoughts)
    
    # Stage 2: Search decisions
    stage2_prompts = []
    for i, question in enumerate(questions):
        stage2_prompts.append(
            f"{SYS_PROMPT}\n\nQuestion: {question}\n"
            f"Thinking content: {thoughts_list[i]}\n"
            "Based on this thinking, is a search for recent information needed?\n"
            "If search is needed, output: <|SEARCH|>\n"
            "If not needed, output nothing\n"
            "Decision:"
        )
    
    # Batch tokenization for stage 2
    stage2_inputs = tokenizer(
        stage2_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024,
        pad_to_multiple_of=8,
        return_token_type_ids=False
    ).to(model.device)
    
    # Batch generation for stage 2
    with torch.no_grad(), torch.cuda.amp.autocast():
        stage2_outputs = model.generate(
            **stage2_inputs,
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Process stage 2 outputs
    for i in range(len(questions)):
        start_index = stage2_inputs.input_ids[i].shape[0]
        search_decision = tokenizer.decode(
            stage2_outputs[i][start_index:], 
            skip_special_tokens=True
        ).strip()
        
        search_token = ""
        if "<|SEARCH|>" in search_decision:
            search_token = " <|SEARCH|>"
        else:
            if any(keyword in thoughts_list[i].lower() for keyword in 
                  ["recent", "current", "202", "real-time", "unknown", "latest", "update"]):
                search_token = " <|SEARCH|>"
        
        final_output = f"{thoughts_list[i]}{search_token}"
        
        batch_data.append({
            "instruction": "Process complex questions and determine search needs",
            "input": questions[i],
            "output": final_output,
            "system": SYS_PROMPT,
            "history": []
        })
    
    return batch_data

# Main function with memory management
def main():
    # Load random SQuAD questions
    squad_questions = load_squad_questions(MAX_QUESTIONS)
    print(f"Loaded {len(squad_questions)} questions")
    
    # Create dataset and dataloader
    dataset = QuestionDataset(squad_questions)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=min(4, os.cpu_count() // 2),  # Optimized workers
        pin_memory=True,
        persistent_workers=True
    )
    
    # Generate dataset
    all_data = []
    
    # Use ThreadPool for parallel batch processing
    with ThreadPoolExecutor(max_workers=NUM_GPUS) as executor:
        futures = []
        for batch in tqdm(dataloader, desc="Processing batches"):
            futures.append(executor.submit(generate_batch, list(batch)))
        
        for future in tqdm(futures, desc="Completing batches"):
            try:
                all_data.extend(future.result())
            except Exception as e:
                print(f"Batch processing failed: {str(e)}")
    
    # Save in Alpaca format
    with open("squad_complex_qa_dataset.json", "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)
    
    print(f"Dataset generation complete. Total samples: {len(all_data)}")

if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 for faster math
    torch.backends.cudnn.benchmark = True  # Enable cuDNN benchmark
    main()