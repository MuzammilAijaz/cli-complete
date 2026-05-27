import sys
import subprocess
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import inquirer
import re

# Constants
MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Use float16 for T4 optimization
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

def clean_command_string(text):
    """
    Targets and scrubs markdown wrappers (```bash, ```, `) and trailing newlines.
    """
    # Remove triple backtick blocks with optional 'bash' tag
    text = re.sub(r"```(?:bash)?\n?", "", text)
    text = re.sub(r"```", "", text)
    # Remove single backticks
    text = re.sub(r"`", "", text)
    # Return stripped single line
    return text.strip().split('\n')[0]

class NL2BashCLI:
    def __init__(self):
        print(f"[*] Initializing {MODEL_NAME} on {DEVICE} (Dtype: {DTYPE})...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=DTYPE,
            device_map="auto"
        )
        self.system_prompt = (
            "You are a specialized Natural Language to Bash translator. "
            "Your task is to convert English requests into a single executable Bash command. "
            "Output ONLY the command. No backticks, no explanations, no markdown, no filler."
        )
        print("[+] System ready.")

    def translate(self, query):
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": query}
        ]
        
        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            generated_ids = self.model.generate(
                model_inputs.input_ids,
                max_new_tokens=64,
                do_sample=False,
                temperature=0.0,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
        # Extract only the newly generated tokens
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
