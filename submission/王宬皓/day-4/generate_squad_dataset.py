from modelscope import AutoModelForCausalLM, AutoTokenizer  # Correct ModelScope imports
import torch
import json
import random
import requests
from tqdm import tqdm

# Load model using ModelScope
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"  # Verified working model in ModelScope
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

# System prompt template
SYS_PROMPT = """You are an AI assistant capable of complex reasoning. Follow this process:
1. Think deeply in the <think> tag
2. Call search when real-time data/unknown knowledge is needed
3. Search call format: <|SEARCH|>"""

# Load SQuAD dataset questions
def load_squad_questions(max_questions=3000):
    try:
        # Load both train and dev sets
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
        
        # Shuffle and select max_questions
        random.shuffle(all_questions)
        return all_questions[:max_questions]
        
    except Exception as e:
        print(f"Error loading SQuAD dataset: {str(e)}")
        # Fallback questions
        return [
            "How does quantum entanglement affect error correction in quantum computers?",
            "Impact of the 2024 global chip shortage on automakers' electrification strategies",
            "Application and limitations of Bayes' theorem in medical diagnosis",
            "Important experiments currently being conducted on the International Space Station?",
            "How does relativity explain time dilation in GPS systems?"
        ]

# Load 3000 random SQuAD questions
squad_questions = load_squad_questions()

# Diversity parameters
diversity_params = {
    "temperature": [0.5, 0.7, 0.9],
    "top_p": [0.8, 0.9, 0.95],
    "repetition_penalty": [1.0, 1.1, 1.2],
    "max_new_tokens": [256, 384, 512]
}

# Multi-stage generation function
def generate_data(question):
    # Stage 1: Generate thinking content
    stage1_prompt = (
        f"Question: {question}\n"
        "Please think deeply. Output format: <think>Your thoughts</think>\n"
        "Thinking:"
    )
    
    # Random diversity parameters
    params = {
        "temperature": random.choice(diversity_params["temperature"]),
        "top_p": random.choice(diversity_params["top_p"]),
        "repetition_penalty": random.choice(diversity_params["repetition_penalty"]),
        "max_new_tokens": random.choice(diversity_params["max_new_tokens"])
    }
    
    # Generate thinking content
    inputs = tokenizer(stage1_prompt, return_tensors="pt").to(model.device)
    stage1_output = model.generate(
        **inputs,
        max_new_tokens=params["max_new_tokens"],
        temperature=params["temperature"],
        top_p=params["top_p"],
        repetition_penalty=params["repetition_penalty"],
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    thoughts = tokenizer.decode(stage1_output[0], skip_special_tokens=True)
    thoughts = thoughts.split("Thinking:")[-1].strip()
    
    # Ensure thoughts are within <think> tags
    if not thoughts.startswith("<think>"):
        thoughts = f"<think>{thoughts}</think>"
    elif not thoughts.endswith("</think>"):
        thoughts = f"{thoughts}</think>"
    
    # Stage 2: Determine if search is needed
    stage2_prompt = (
        f"{SYS_PROMPT}\n\nQuestion: {question}\n"
        f"Thinking content: {thoughts}\n"
        "Based on this thinking, is a search for recent information needed?\n"
        "If search is needed, output: <|SEARCH|>\n"
        "If not needed, output nothing\n"
        "Decision:"
    )
    
    inputs = tokenizer(stage2_prompt, return_tensors="pt").to(model.device)
    stage2_output = model.generate(
        **inputs,
        max_new_tokens=10,
        temperature=0.3,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
    
    search_decision = tokenizer.decode(stage2_output[0], skip_special_tokens=True)
    search_decision = search_decision.split("Decision:")[-1].strip()
    
    # Process search decision
    search_token = ""
    if "<|SEARCH|>" in search_decision:
        search_token = " <|SEARCH|>"
    else:
        # Validate decision - add search if thoughts mention recent data
        if any(keyword in thoughts.lower() for keyword in ["recent", "current", "202", "real-time", "unknown", "latest", "update"]):
            search_token = " <|SEARCH|>"
    
    # Final output
    final_output = f"{thoughts}{search_token}"
    
    return {
        "instruction": "Process complex questions and determine search needs",
        "input": question,
        "output": final_output,
        "system": SYS_PROMPT,
        "history": []
    }

# Generate dataset
dataset = []
for question in tqdm(squad_questions, desc="Generating dataset"):
    try:
        data = generate_data(question)
        dataset.append(data)
    except Exception as e:
        print(f"Error processing: {question[:50]}... - {str(e)}")
    
# Save in Alpaca format
with open("squad_complex_qa_dataset.json", "w", encoding="utf-8") as f:
    json.dump(dataset, f, ensure_ascii=False, indent=2)

print(f"Dataset generation complete. Total samples: {len(dataset)}")