import sys
import subprocess
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model, PeftModel
from trl import SFTConfig, SFTTrainer
from datasets import Dataset
import inquirer
import re
import os

# Constants
MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
ADAPTER_PATH = "./qwen-nl2bash-adapter"

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU required for training.")
DEVICE = "cuda"
# Use float16 for T4 optimization
# DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32
 DTYPE = torch.float16

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
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=DTYPE,
            device_map="auto"
        )
        
        if os.path.exists(ADAPTER_PATH):
            print(f"[*] Loading fine-tuned adapter from {ADAPTER_PATH}...")
            self.model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
        else:
            self.model = base_model
            
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
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            generated_ids = self.model.generate(
                model_inputs.input_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                temperature=0.0,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        raw_response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return clean_command_string(raw_response)

    def fine_tune(self, data_dir="data/bash"):
        """
        Performs local fine-tuning using LoRA on the NL2Bash dataset.
        """
        print("[*] Preparing data for fine-tuning...")
        nl_path = os.path.join(data_dir, "all.nl")
        cm_path = os.path.join(data_dir, "all.cm")
        
        if not os.path.exists(nl_path) or not os.path.exists(cm_path):
            print(f"[x] Error: Data files not found in {data_dir}")
            return

        with open(nl_path, "r", encoding="utf-8") as f_nl, open(cm_path, "r", encoding="utf-8") as f_cm:
            nls = f_nl.readlines()
            cms = f_cm.readlines()
        
        formatted_data = []
        for nl, cm in zip(nls, cms):
            nl, cm = nl.strip(), cm.strip()
            if len(nl.split()) > 50: continue # Filter long descriptions
            
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": nl},
                {"role": "assistant", "content": cm}
            ]
            text = self.tokenizer.apply_chat_template(messages, tokenize=False)
            formatted_data.append({"text": text})

        dataset = Dataset.from_list(formatted_data)
        
        # ==== Configuration ==========================================================
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        print("[*] Starting SFT Training (LoRA)...")
        training_args = SFTConfig(
            output_dir="./qwen-sft-results",
            max_length=512,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            num_train_epochs=1,
            logging_steps=10,
            save_steps=100,
            bf16=False, # Use float16
            fp16=torch.cuda.is_available(),
            dataset_text_field="text"
        )
        
        trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            peft_config=lora_config,
        )
        # =============================================================================
        
        trainer.train()
        self.model.save_pretrained(ADAPTER_PATH)
        print(f"[+] Training complete. Adapter saved to {ADAPTER_PATH}")

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
        print("Type 'exit' to quit, 'train' to fine-tune.\n")

        while True:
            try:
                query = input("nl2bash> ").strip()
                if not query or query.lower() == "exit":
                    break
                
                if query.lower() == "train":
                    confirm = [inquirer.Confirm('confirm', message="Start local fine-tuning?", default=False)]
                    if inquirer.prompt(confirm)['confirm']:
                        self.fine_tune()
                    continue

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
