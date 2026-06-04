import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import os
import re
from collections import Counter
import tomllib

# Default Constants (can be overridden or imported)
try:
    with open("configs/training.toml", "rb") as f:
        config = tomllib.load(f)
    MODEL_NAME = config["model"]["name"]
except:
    MODEL_NAME = "Qwen/Qwen2.5-Coder-0.5B-Instruct"

MERGED_PATH = "./qwen-nl2bash-merged"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32
TEST_NL_PATH = "data/bash/custom_test.nl"
TEST_CM_PATH = "data/bash/custom_test.cm" # Fixed from test.cm

def load_test_data(limit=50, nl_path=TEST_NL_PATH, cm_path=TEST_CM_PATH):
    if not os.path.exists(nl_path) or not os.path.exists(cm_path):
        print(f"[!] Test dataset files not found: {nl_path} or {cm_path}")
        return []
    
    with open(nl_path, "r", encoding="utf-8") as f_nl, open(
            cm_path, "r", encoding="utf-8") as f_cm:
        nls = [line.strip() for line in f_nl.readlines()]
        cms = [line.strip() for line in f_cm.readlines()]
    
    return list(zip(nls, cms))[:limit]

def clean_command_string(text):
    """Targets and scrubs markdown wrappers and trailing newlines."""
    text = re.sub(r"```(?:bash)?\n?", "", text)
    text = re.sub(r"```", "", text)
    text = re.sub(r"`", "", text)
    # Extract only the first command before any irrelevant pipes added by hallucination
    # if the gold standard doesn't have a pipe but prediction does, we check if it's filler
    return text.strip().split('\n')[0]

def get_tokens(cmd):
    """Simple tokenizer for bash commands."""
    return re.findall(r'\w+|[^\w\s]', cmd.lower())

def calculate_metrics(pred, gold):
    """
    Calculates a suite of NLP metrics for Bash commands.
    """
    p_tokens = get_tokens(pred)
    g_tokens = get_tokens(gold)
    
    if not g_tokens: return 0, 0, 0, 0
    
    # 1. Strict EM
    em = 1.0 if pred.strip().lower() == gold.strip().lower() else 0.0
    
    # 2. Utility Match (First token)
    u_match = 1.0 if (p_tokens and g_tokens and p_tokens[0] == g_tokens[0]) else 0.0
    
    # 3. Keyword Recall (Arguments/Paths - tokens that aren't flags or common utils)
    # We define keywords as tokens that don't start with '-' and aren't symbols
    keywords = [t for t in g_tokens if not t.startswith('-') and t.isalnum() and len(t) > 2]
    if not keywords:
        kw_recall = 1.0 # default if no keywords
    else:
        found = sum(1 for kw in keywords if kw in p_tokens)
        kw_recall = found / len(keywords)
        
    # 4. Token F1
    common = Counter(p_tokens) & Counter(g_tokens)
    num_same = sum(common.values())
    if len(p_tokens) == 0 or len(g_tokens) == 0:
        f1 = 0.0
    else:
        precision = num_same / len(p_tokens)
        recall = num_same / len(g_tokens)
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        
    return em, u_match, kw_recall, f1

class Evaluator:
    def __init__(self):
        # Prefer merged model if available
        load_path = MERGED_PATH if os.path.exists(MERGED_PATH) else MODEL_NAME
        print(f"[*] Loading model from {load_path} for evaluation...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(load_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            load_path,
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
        metrics_list = []
        
        for nl, expected in data:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": nl}
            ]
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

            start = time.perf_counter()
            with torch.inference_mode():
                output_ids = self.model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=64,
                    num_beams=num_beams,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            latency = (time.perf_counter() - start) * 1000
            latencies.append(latency)

            # Extract generated text
            generated_ids = output_ids[0][len(inputs.input_ids[0]):]
            raw_predicted = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            predicted = clean_command_string(raw_predicted)
            
            # Calculate flexible metrics
            m = calculate_metrics(predicted, expected)
            metrics_list.append(m)
            
            # if len(latencies) <= 3:
            print(f"  [NL]: {nl}")
            print(f"  [Pred]: {predicted}")
            print(f"  [Gold]: {expected}")
            print(f"  [Score]: EM:{m[0]} Util:{m[1]} KW-Rec:{m[2]:.2f} F1:{m[3]:.2f}\n")

        avg_latency = np.mean(latencies)
        m_avgs = np.mean(metrics_list, axis=0) * 100
        
        return {
            "strategy": strategy_name, 
            "latency": avg_latency, 
            "em": m_avgs[0], 
            "util": m_avgs[1],
            "kw_rec": m_avgs[2],
            "f1": m_avgs[3]
        }

if __name__ == "__main__":

    # Load samples
    test_subset = load_test_data(15)
    if not test_subset:
        print("[x] Error: No test data found. Run setup first.")
        exit(1)

    evaluator = Evaluator()
    
    # 1. Warmup (ignore results)
    evaluator.evaluate_strategy(test_subset[:1], "Warmup", num_beams=1)
    
    # 2. Greedy Decoding
    greedy_results = evaluator.evaluate_strategy(test_subset, "Greedy Decoding", num_beams=1)
    
    # 3. Beam Search
    beam_results = evaluator.evaluate_strategy(test_subset, "3-Beam Search", num_beams=3)

    print("\n### Performance Comparison Matrix (NL2Bash Test Split)\n")
    print("| Strategy | Latency (ms) | EM % | Util Match % | KW Recall % | Token F1 % |")
    print("|----------|--------------|------|--------------|-------------|------------|")
    for res in [greedy_results, beam_results]:
        print(f"| {res['strategy']:<15} | {res['latency']:>12.2f} | {res['em']:>4.1f}% | {res['util']:>12.1f}% | {res['kw_rec']:>11.1f}% | {res['f1']:>10.1f}% |")
    print("\n")
    print("Note: 'KW Recall' measures the retention of critical arguments and file paths.")
    print("      'Util Match' measures if the primary tool (e.g., find, tar, ls) was correctly selected.")
