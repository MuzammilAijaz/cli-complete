import sys
import subprocess
import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration
import inquirer

# Constants
MODEL_NAME = "Salesforce/codet5-small"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Local fallback dictionary for high-stakes/common queries
FALLBACKS = {
    "convert folder to zip": "zip -r {0}.zip {0}",
    "zip folder": "zip -r {0}.zip {0}",
    "extract zip": "unzip {0}",
    "list files": "ls -la",
    "search text in files": "grep -rnw '.' -e '{0}'",
    "find file by name": "find . -name '{0}'",
}

class NL2BashCLI:
    def __init__(self):
        print(f"[*] Loading model {MODEL_NAME} on {DEVICE}...")
        # Override additional_special_tokens to avoid TypeError in transformers 5.9.0
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, additional_special_tokens=[])
        self.model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME).to(DEVICE)
        print("[+] Model loaded successfully.")

    def translate(self, query, num_return_sequences=3):
        # Check fallbacks first (simple heuristic)
        for key, template in FALLBACKS.items():
            if key in query.lower():
                # This is a bit naive but serves as a "smart fallback catcher"
                # For demonstration purposes, we'll just return the template if it matches roughly
                # Real implementation might need more sophisticated regex
                return [template.format("target")]

        input_text = f"translate English to Bash: {query}"
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(DEVICE)

        outputs = self.model.generate(
            input_ids,
            max_length=64,
            num_beams=5,
            num_return_sequences=num_return_sequences,
            early_stopping=True
        )

        return [self.tokenizer.decode(out, skip_special_tokens=True) for out in outputs]

    def execute_command(self, cmd):
        print(f"\n[!] Executing: {cmd}")
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=False)
            if result.stdout:
                print("--- Output ---")
                print(result.stdout)
            if result.stderr:
                print("--- Error ---")
                print(result.stderr)
            print(f"--- Return Code: {result.returncode} ---")
        except Exception as e:
            print(f"[x] Execution failed: {e}")

    def run(self):
        print("\n--- NL2Bash Local CLI ---")
        print("Type 'exit' or 'quit' to stop.\n")

        while True:
            try:
                query = input("nl2bash> ").strip()
                if not query:
                    continue
                if query.lower() in ["exit", "quit"]:
                    break

                results = self.translate(query)
                
                if not results:
                    print("[-] No translations found.")
                    continue

                # Deduplicate and format choices
                choices = list(dict.fromkeys(results))
                choices.append("Cancel")

                questions = [
                    inquirer.List('command',
                                  message="Select a command to execute",
                                  choices=choices,
                                  ),
                ]
                
                answers = inquirer.prompt(questions)
                if not answers or answers['command'] == "Cancel":
                    continue

                selected_cmd = answers['command']
                
                confirm = [
                    inquirer.Confirm('confirm',
                                     message=f"Are you sure you want to run: {selected_cmd}?",
                                     default=False),
                ]
                
                if inquirer.prompt(confirm)['confirm']:
                    self.execute_command(selected_cmd)

            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"[x] An error occurred: {e}")

if __name__ == "__main__":
    cli = NL2BashCLI()
    cli.run()
