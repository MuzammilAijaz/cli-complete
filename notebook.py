# %% [markdown]
# # NL2Bash: Efficient Natural Language to Bash Translation
# 
# ## Project Overview
# This notebook serves as the primary **computable document** for the NL2Bash project. It bridges the gap between the high-level research objectives documented in our IEEE term paper and the practical implementation of a local, privacy-preserving terminal assistant.
# 
# ### The Problem
# The command line is an extremely powerful interface, but its steep learning curve often hinders productivity. Translating Natural Language (NL) to Bash is difficult because:
# 1. **Structural Complexity:** Bash commands often involve pipes, subshells, and complex regex.
# 2. **Token Variability:** Rare tokens like file paths and specific flags are hard for general models to predict.
# 3. **Safety Risks:** Unlike Text-to-SQL which operates on a database, Bash operates on the host operating system, making safety paramount.
# 
# ### Our Methodology
# We utilize a **Small Language Model (SLM)** strategy. Instead of relying on multi-billion parameter cloud models, we fine-tune a 0.5B parameter model (**Qwen2.5-Coder**) using **Low-Rank Adaptation (LoRA)**. This allows us to achieve high-performance domain adaptation while keeping the model small enough to run on a standard CPU.

# %% [markdown]
# ## 0. Environment Setup (Google Colab)
# In this section, we prepare the virtual environment. Following modern best practices, we use `uv` for dependency management. 
# 
# **Why uv?** As noted in our research, traditional `pip` can be slow and prone to dependency conflicts in Colab. `uv` provides a fast, Rust-based resolver that ensures our environment matches the `pyproject.toml` specification exactly.

# %%
import os

# 1. Clean start to ensure no path conflicts
%cd /content
if os.path.exists('cli-complete'):
    !rm -rf cli-complete

# 2. Install uv (modern, fast package manager)
!pip install -q uv

# 3. Clone the project from GitHub
!git clone https://github.com/MuzammilAijaz/cli-complete.git
%cd cli-complete

# 4. Sync the environment
# We install the project in 'editable' mode using the system python.
!uv pip install --system .

# %% [markdown]
# ## 1. System Configuration & Hardware Detection
# Our architecture is designed to be **hardware-agnostic**. The code below detects if a CUDA GPU is available; if not, it falls back to CPU-optimized BFloat16 or Float32 inference.
# 
# ### Global Constants
# We load parameters from `configs/training.toml` to ensure consistency between the training pipeline and the inference engine.

# %%
import torch
import tomllib
import os

# Load configuration from the shared TOML file
config_path = "configs/training.toml"
if os.path.exists(config_path):
    with open(config_path, "rb") as f:
        config = tomllib.load(f)
else:
    print("[!] Warning: training.toml not found, using default constants.")
    config = {"model": {"name": "Qwen/Qwen2.5-Coder-0.5B-Instruct"}}

# Import symbols from our main engine
from main import DEVICE, DTYPE, MODEL_NAME, ADAPTER_PATH

print(f"[*] Researching Model: {MODEL_NAME}")
print(f"[*] Computation Device: {DEVICE}")
print(f"[*] Tensor Precision: {DTYPE}")

# %% [markdown]
# ## 2. The Neural Engine: Model Loading & Merging
# The "Neural Engine" layer of our architecture employs a **Multi-Stage Loading Strategy**. 
# 
# 1. **Merged Model (Fast Path):** If we have previously merged the LoRA adapter into the base model, we load it directly. This is the "production" mode.
# 2. **Base + Adapter (Development Path):** If we have a fresh adapter from a training run, we load the base weights, attach the LoRA matrices, and merge them in-memory.
# 3. **Zero-Shot Fallback:** If no specialized weights are found, the model runs in its pre-trained state.

# %%
from main import NL2BashCLI

# The NL2BashCLI class encapsulates our multi-stage loading logic
cli = NL2BashCLI()
tokenizer, model = cli.tokenizer, cli.model

# %% [markdown]
# ## 3. Interactive Prototyping Sandbox
# This section allows for qualitative evaluation. While quantitative benchmarks tell us about accuracy, qualitative testing helps us understand the "personality" and reasoning of the model.
# 
# ### Methodology: Prompt Engineering
# We use a specialized system prompt that instructs the model to act as a focused translator. We discourage explanations to keep the output "terminal-ready."

# %%
def translate_and_display(query):
    print(f"Input Query: {query}")
    start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    
    # Run the neural translation
    command = cli.translate(query)
    
    print(f"Neural Suggestion: {command}")
    print("-" * 30)

# Test a variety of common terminal intents
test_queries = [
    "find all log files modified in the last 7 days",
    "check which processes are using port 8080",
    "extract backup.tar.gz to the /tmp folder",
    "show the disk usage of every directory in the current path",
    "replace the string 'error' with 'warning' in all .txt files"
]

print("--- Interactive Sandbox Output ---")
for query in test_queries:
    translate_and_display(query)

# %% [markdown]
# ## 4. Quantitative Benchmarking
# To measure our progress scientifically, we evaluate the model against a "Gold Standard" test set.
# 
# ### Evaluation Metrics
# 1. **Exact Match (EM):** Does the command match the human expert's command exactly? (Very Strict)
# 2. **Utility Match:** Was the primary tool (e.g., `grep`, `find`) correctly identified?
# 3. **Token F1:** Measures the overlap of flags and arguments between the prediction and the truth.
# 
# > **Note on Overfitting:** As discussed in our paper, if you are using `all.nl` for testing, the results will be overly optimistic. Proper research requires the `test.nl` split.

# %%
from eval_benchmark import Evaluator, load_test_data

# Load a small batch for demonstration
test_samples = load_test_data(limit=10)

if test_samples:
    evaluator = Evaluator()
    evaluator.tokenizer = tokenizer
    evaluator.model = model
    
    # Evaluate using Greedy Decoding (fastest, most reproducible)
    results = evaluator.evaluate_strategy(test_samples, "Greedy Benchmark", num_beams=1)
    
    print("\n### Performance Report")
    print(f"| Metric       | Performance | Description |")
    print(f"|--------------|-------------|-------------|")
    print(f"| Exact Match  | {results['em']:.1f}%      | Identity with Gold Standard |")
    print(f"| Utility Match| {results['util']:.1f}%      | Core tool selection |")
    print(f"| Token F1     | {results['f1']:.1f}%      | Flag/Argument retention |")
else:
    print("[!] Test dataset not found in data/bash/. Please check project setup.")

# %% [markdown]
# ## 5. Training Phase: Fine-Tuning with LoRA
# This is where the model learns the specific "language" of Bash.
# 
# ### What is SFT?
# Supervised Fine-Tuning (SFT) teaches the model to follow a specific output format. By using the NL2Bash dataset, we "bias" the model towards correct Bash syntax while retaining its general knowledge.
# 
# ### What is LoRA?
# Instead of updating all 500 million parameters, we only update a tiny fraction (the LoRA adapters). This is why we can fine-tune this model even on a single consumer GPU.

# %%
from train import NL2BashTrainer

if torch.cuda.is_available():
    print("[*] High-performance hardware (GPU) detected. Training is available.")
    # To start training, uncomment the lines below. 
    # Warning: This will take several minutes and create a new adapter in qwen-nl2bash-adapter/
    
    # trainer = NL2BashTrainer()
    # trainer.fine_tune()
else:
    print("[!] No GPU detected. Training is disabled to prevent system hang.")

# %% [markdown]
# ## 6. Conclusion & Future Directions
# Our current prototype successfully demonstrates that a **0.5B SLM** is capable of meaningful Bash translation on local hardware.
# 
# ### Identified Improvements
# 1. **Safety Layer:** Transitioning from model-based safety to a **Systems Engineering** approach using JSON-parsing of risky utilities.
# 2. **Evaluation Rigor:** Integrating execution-based testing (does the command actually run and produce the right files?).
# 3. **Speed:** Implementing GGUF quantization for even faster CPU performance.
