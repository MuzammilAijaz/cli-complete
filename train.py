import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer
from datasets import Dataset
import re
import os
import tomllib
import json

# Load configuration
with open("configs/training.toml", "rb") as f:
    config = tomllib.load(f)

# Constants
MODEL_NAME = config["model"]["name"]
ADAPTER_PATH = config["model"]["adapter_path"]

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU required for training.")
DEVICE = "cuda"
DTYPE = torch.float16

class NL2BashTrainer:
    def __init__(self):
        print(f"[*] Initializing {MODEL_NAME} for training on {DEVICE}...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
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

            # filter long descriptions
            max_words = config["data"]["max_nl_words"]
            if len(nl.split()) > max_words:
                continue
            
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": nl},
                {"role": "assistant", "content": cm}
            ]
            text = self.tokenizer.apply_chat_template(messages, tokenize=False)
            formatted_data.append({"text": text})

        dataset = Dataset.from_list(formatted_data)
        
        lora_cfg = config["lora"]
        lora_config = LoraConfig(
                r=lora_cfg["r"],
                lora_alpha=lora_cfg["alpha"],
                target_modules=lora_cfg["target_modules"],
                lora_dropout=lora_cfg["dropout"],
                bias="none",
                task_type="CAUSAL_LM"
        )

        print("[*] Starting SFT Training (LoRA)...")

        train_cfg = config["training"]
        training_args = SFTConfig(
                output_dir="./qwen-sft-results",
                max_seq_length=train_cfg["max_length"],
                per_device_train_batch_size=train_cfg["batch_size"],
                gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
                learning_rate=train_cfg["learning_rate"],
                num_train_epochs=train_cfg["epochs"],
                logging_steps=train_cfg["logging_steps"],
                save_steps=train_cfg["save_steps"],
                fp16=train_cfg["fp16"],
                bf16=train_cfg["bf16"],
                gradient_checkpointing=True, 
                dataset_text_field="text"
        )

        trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            peft_config=lora_config,
        )
        
        trainer.train()

        # Save model + config
        trainer.model.save_pretrained(ADAPTER_PATH)
        training_args.save_pretrained(ADAPTER_PATH)
        lora_config.save_pretrained(ADAPTER_PATH)
        print(f"[+] Training complete. Adapter and hyperparameters saved to {ADAPTER_PATH}")

if __name__ == "__main__":
    trainer = NL2BashTrainer()
    trainer.fine_tune()
