import time
import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration
import numpy as np

# Constants
MODEL_NAME = "Salesforce/codet5-small"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Mock dataset subset (NL, Expected Bash)
TEST_DATA = [
    ("list all files in the current directory", "ls"),
    ("list all files including hidden ones", "ls -a"),
    ("find files named 'test.txt' in current directory", "find . -name 'test.txt'"),
    ("copy file.txt to /tmp", "cp file.txt /tmp"),
    ("remove the directory 'data' and all its contents", "rm -rf data"),
    ("show disk usage for the current folder", "du -sh ."),
    ("search for 'error' in log.txt", "grep error log.txt"),
    ("change permissions of script.sh to executable", "chmod +x script.sh"),
    ("create a new directory named 'backup'", "mkdir backup"),
    ("print the first 10 lines of file.txt", "head -n 10 file.txt")
]

class Benchmark:
    def __init__(self):
        print(f"[*] Loading model {MODEL_NAME} for benchmark...")
        # Override additional_special_tokens to avoid TypeError in transformers 5.9.0
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, additional_special_tokens=[])
        self.model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME).to(DEVICE)
        print("[+] Model loaded.")

    def evaluate(self, strategy_name, num_beams):
        print(f"\n[*] Evaluating Strategy: {strategy_name} (Beams: {num_beams})...")
        latencies = []
        matches = 0
        total = len(TEST_DATA)

        for nl, expected in TEST_DATA:
            input_text = f"translate English to Bash: {nl}"
            input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(DEVICE)

            start_time = time.perf_counter()
            with torch.no_grad():
                output = self.model.generate(
                    input_ids,
                    max_length=64,
                    num_beams=num_beams,
                    num_return_sequences=1,
                    early_stopping=True
                )
            latency = (time.perf_counter() - start_time) * 1000  # ms
            latencies.append(latency)

            predicted = self.tokenizer.decode(output[0], skip_special_tokens=True).strip()
            
            if predicted == expected:
                matches += 1
            
            # Debug output for first few
            if len(latencies) <= 3:
                print(f"  [NL]: {nl}")
                print(f"  [Pred]: {predicted} | [Exp]: {expected}")

        avg_latency = np.mean(latencies)
        em_accuracy = (matches / total) * 100
        
        return {
            "strategy": strategy_name,
            "avg_latency": avg_latency,
            "em_accuracy": em_accuracy
        }

def print_results_table(results):
    print("\n### Experimental Results: NL2Bash Translation Performance\n")
    print("| Decoding Strategy | Avg Latency (ms) | Exact Match (EM) % |")
    print("|-------------------|-------------------|--------------------|")
    for res in results:
        print(f"| {res['strategy']:<17} | {res['avg_latency']:>17.2f} | {res['em_accuracy']:>18.1f}% |")
    print("\n")

if __name__ == "__main__":
    bench = Benchmark()
    
    # Warmup
    print("[*] Performing warmup...")
    bench.evaluate("Warmup", 1)
    
    # Actual Benchmark
    greedy_results = bench.evaluate("Greedy Decoding", 1)
    beam_results = bench.evaluate("5-Beam Search", 5)
    
    print_results_table([greedy_results, beam_results])
