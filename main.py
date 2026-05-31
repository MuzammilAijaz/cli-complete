import sys
import subprocess
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import inquirer
import re
import os
import tomllib

# Load configuration
with open("configs/training.toml", "rb") as f:
    config = tomllib.load(f)

# Constants
MODEL_NAME = config["model"]["name"]
ADAPTER_PATH = config["model"]["adapter_path"]
MERGED_PATH = "./qwen-nl2bash-merged"

# Dynamic hardware detection
if torch.cuda.is_available():
    DEVICE = "cuda"
    DTYPE = torch.float16
    print(f"[*] CUDA detected. Using {DEVICE} with {DTYPE}")
else:
    DEVICE = "cpu"
    # Use bfloat16 on CPU if supported for speed, else float32
    DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    print(f"[*] CUDA not found. Falling back to {DEVICE} with {DTYPE}")

MAX_NEW_TOKENS = 32

def clean_command_string(text):
    """
    Targets and scrubs markdown wrappers (```bash, ```, `) and trailing newlines.
    """
    text = re.sub(r"```(?:bash)?\n?", "", text)
    text = re.sub(r"```", "", text)
    text = re.sub(r"`", "", text)
    return text.strip().split('\n')[0]

class NL2BashCLI:
    def __init__(self):
        print(f"[*] Initializing {MODEL_NAME} on {DEVICE}...")
        self.tokenizer, self.model = self.load_model()
        self.system_prompt = (
            "You are a specialized Natural Language to Bash translator. "
            "Your task is to convert English requests into a single executable Bash command. "
            "Output ONLY the command. No backticks, no explanations, no markdown, no filler."
        )
        print("[+] System ready.")

    def load_model(self):
        """
        Adopts the methodology from run_fast.py:
        Check for merged model, if not exists merge and save.
        """
        if os.path.exists(MERGED_PATH):
            print("[+] Found merged model. Loading...")
            tokenizer = AutoTokenizer.from_pretrained(MERGED_PATH)
            model = AutoModelForCausalLM.from_pretrained(
                MERGED_PATH,
                torch_dtype=DTYPE,
                device_map="auto"
            )
            return tokenizer, model

        if not os.path.exists(ADAPTER_PATH):
            print("[!] No adapter found. Loading base model only.")
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                torch_dtype=DTYPE,
                device_map="auto"
            )
            return tokenizer, model

        print("[+] No merged model found. Loading base model and adapter...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=DTYPE,
            device_map="auto"
        )
        peft_model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

        print("[+] Merging adapter into base model...")
        merged_model = peft_model.merge_and_unload()

        print("[+] Saving merged model for future fast loading...")
        merged_model.save_pretrained(MERGED_PATH)
        tokenizer.save_pretrained(MERGED_PATH)
        print("[+] Merged model saved.")

        return tokenizer, merged_model

    def translate(self, query):
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": query}
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        with torch.inference_mode():
            generated_ids = self.model.generate(
                model_inputs.input_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                eos_token_id=self.tokenizer.eos_token_id,
                do_sample=False,
                temperature=0.0,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        raw_response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return clean_command_string(raw_response)

    def execute_command(self, cmd):
        print(f"\n[!] Executing: {cmd}")
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
            if result.stdout:
                print("--- Output ---")
                print(result.stdout)
            if result.stderr:
                print("--- Error ---")
                print(result.stderr)
            print(f"--- Return Code: {result.returncode} ---")
        except subprocess.TimeoutExpired:
            print("[x] Execution timed out.")
        except Exception as e:
            print(f"[x] Execution failed: {e}")

    def run(self):
        print("\n=== NL2Bash Local CLI (Qwen2.5-Coder) ===")
        print("Fast, 100% Local Semantic Parsing.")
        print("Type 'exit' to quit.\n")

        while True:
            try:
                query = input("nl2bash> ").strip()
                if not query or query.lower() == "exit":
                    break

                # Generate command
                command = self.translate(query)
                
                if not command:
                    print("[-] No command generated.")
                    continue

                # Prompt for execution
                confirm = [
                    inquirer.Confirm('confirm',
                                     message=f"Generated: `{command}`. Run it?",
                                     default=False),
                ]
                
                result = inquirer.prompt(confirm)
                if result and result['confirm']:
                    self.execute_command(command)
                else:
                    print("[*] Aborted.")

            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"[x] Error: {e}")

if __name__ == "__main__":
    cli = NL2BashCLI()
    cli.run()
