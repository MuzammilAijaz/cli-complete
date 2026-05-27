import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import os
import re
from collections import Counter

# Constants
MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32
TRAIN_NL_PATH = "data/bash/train.nl"
TRAIN_CM_PATH = "data/bash/train.cm"

def load_test_data(limit=10):
    if not os.path.exists(TRAIN_NL_PATH) or not os.path.exists(TRAIN_CM_PATH):
        print("[!] Dataset files not found.")
        return []
    
    with open(TRAIN_NL_PATH, "r") as f_nl, open(TRAIN_CM_PATH, "r") as f_cm:
        nls = [line.strip() for line in f_nl.readlines()]
        cms = [line.strip() for line in f_cm.readlines()]
    
    return list(zip(nls, cms))[:limit]

def clean_command_string(text):
    """Targets and scrubs markdown wrappers and trailing newlines."""
    text = re.sub(r"```(?:bash)?\n?", "", text)
    text = re.sub(r"```", "", text)
    text = re.sub(r"`", "", text)
    return text.strip().split('\n')[0]

def calculate_f1(pred, gold):
    """Calculates token-level F1 score."""
    pred_tokens = re.findall(r'\w+|[^\w\s]', pred.lower())
    gold_tokens = re.findall(r'\w+|[^\w\s]', gold.lower())
    
    if not pred_tokens or not gold_tokens:
        return 0.0
    
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * precision * recall / (precision + recall)
    return f1

class Evaluator:
    def __init__(self):
        print(f"[*] Loading model {MODEL_NAME} for evaluation (Dtype: {DTYPE})...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=DTYPE,
            device_map="auto"
        )
        self.system_prompt = (
            "You are a specialized Natural Language to Bash translator. "
            "Output ONLY the single-line executable bash command. No markdown, no filler."
        )

    def evaluate_strategy(self, data, strategy_name, num_beams=1):
        print(f"\n[*] Running Benchmark: {strategy_name}...")
        latencies = []
        em_matches = 0
        f1_scores = []
        
        for nl, expected in data:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": nl}
            ]
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer([text], return_tensors="pt").to(DEVICE)

            start = time.perf_counter()
            with torch.no_grad():
                output_ids = self.model.generate(
                    inputs.input_ids,
                    max_new_tokens=64,
                    num_beams=num_beams,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            latency = (time.perf_counter() - start) * 1000
            latencies.append(latency)

            # Extract generated text
            generated_ids = output_ids[0][len(inputs.input_ids[0]):]
            raw_predicted = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            predicted = clean_command_string(raw_predicted)
            
            # Metrics
            if predicted.lower() == expected.lower():
                em_matches += 1
            
            f1_scores.append(calculate_f1(predicted, expected))
            
            if len(latencies) <= 2:
                print(f"  [NL]: {nl}")
                print(f"  [Pred]: {predicted} | [Exp]: {expected}")

        avg_latency = np.mean(latencies)
        accuracy = (em_matches / len(data)) * 100
        avg_f1 = np.mean(f1_scores) * 100
        
        return {
            "strategy": strategy_name, 
            "latency": avg_latency, 
            "accuracy": accuracy, 
            "f1": avg_f1
        }

if __name__ == "__main__":
    test_subset = load_test_data(10)
    if not test_subset:
        sys.exit(1)

    evaluator = Evaluator()
    
    # 1. Warmup
    evaluator.evaluate_strategy(test_subset[:1], "Warmup", num_beams=1)
    
    # 2. Greedy Decoding
    greedy_results = evaluator.evaluate_strategy(test_subset, "Greedy Decoding", num_beams=1)
    
    # 3. Beam Search
    beam_results = evaluator.evaluate_strategy(test_subset, "3-Beam Search", num_beams=3)

    print("\n### Performance Comparison Matrix\n")
    print("| Strategy | Avg Latency (ms) | EM Accuracy % | Token F1 % |")
    print("|----------|------------------|---------------|------------|")
    for res in [greedy_results, beam_results]:
        print(f"| {res['strategy']:<15} | {res['latency']:>16.2f} | {res['accuracy']:>12.1f}% | {res['f1']:>9.1f}% |")
    print("\n")
