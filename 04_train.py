"""
MedBridge — QLoRA Fine-Tuning Script
======================================
Model  : meta-llama/Meta-Llama-3.1-8B-Instruct  (or Mistral-7B-Instruct-v0.3)
Method : QLoRA — 4-bit NF4 quantization + LoRA adapters via PEFT
Trainer: TRL SFTTrainer with chat-template formatting
Logging: Weights & Biases

Requirements:
    pip install torch transformers==4.43.0 datasets peft trl \
                bitsandbytes accelerate wandb flash-attn

GPU memory requirements:
    QLoRA 4-bit : ~14 GB VRAM  (RTX 3090 / A10G / T4×2 / A100)
    Full BF16   : ~60 GB VRAM  (A100 80GB — not needed here)

Usage:
    # Single GPU
    python 04_train.py

    # Multi-GPU (2× A100 etc.)
    torchrun --nproc_per_node=2 04_train.py
"""

import os, math
from pathlib import Path
from dataclasses import dataclass, field

import torch
import wandb
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    set_seed,
)
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM


# ══════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class MedBridgeConfig:
    # ── Model ──────────────────────────────────────────────────────────────
    model_id: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    # Alternatives (uncomment to swap):
    # model_id: str = "mistralai/Mistral-7B-Instruct-v0.3"
    # model_id: str = "google/gemma-2-9b-it"

    # ── Data ───────────────────────────────────────────────────────────────
    data_root: str = "data"
    output_dir: str = "outputs/medbridge-v1"

    # ── QLoRA ──────────────────────────────────────────────────────────────
    lora_r: int          = 16      # rank — 16 is sweet spot for med tasks
    lora_alpha: int      = 32      # always 2× rank
    lora_dropout: float  = 0.05
    # Targets: all attention projections + feed-forward gates
    lora_target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",   # MLP layers — important for knowledge
    ])

    # ── Training ───────────────────────────────────────────────────────────
    num_epochs: int                  = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4     # effective batch = 16
    learning_rate: float             = 2e-4
    lr_scheduler_type: str           = "cosine"
    warmup_ratio: float              = 0.05
    max_seq_length: int              = 1024
    weight_decay: float              = 0.01

    # ── Logging / saving ───────────────────────────────────────────────────
    logging_steps: int  = 10
    eval_steps: int     = 200
    save_steps: int     = 400
    save_total_limit: int = 3

    # ── Misc ───────────────────────────────────────────────────────────────
    seed: int = 42
    bf16: bool = True     # Use BF16 on Ampere+. Set False + fp16=True for T4.
    fp16: bool = False


CFG = MedBridgeConfig()


# ══════════════════════════════════════════════════════════════════════════
# WEIGHTS & BIASES
# ══════════════════════════════════════════════════════════════════════════

wandb.init(
    project="medbridge-llm",
    name=f"qlora-{CFG.model_id.split('/')[-1]}-r{CFG.lora_r}",
    config={
        "model":        CFG.model_id,
        "lora_r":       CFG.lora_r,
        "lora_alpha":   CFG.lora_alpha,
        "lr":           CFG.learning_rate,
        "epochs":       CFG.num_epochs,
        "eff_batch":    CFG.per_device_train_batch_size * CFG.gradient_accumulation_steps,
        "max_seq_len":  CFG.max_seq_length,
        "dataset":      "alphacare-52k-filtered",
        "train_n":      20905,
    }
)

set_seed(CFG.seed)


# ══════════════════════════════════════════════════════════════════════════
# 1. TOKENIZER
# ══════════════════════════════════════════════════════════════════════════

print("\n[1] Loading tokenizer …")
tokenizer = AutoTokenizer.from_pretrained(
    CFG.model_id,
    trust_remote_code=True,
)
tokenizer.pad_token     = tokenizer.eos_token
tokenizer.padding_side  = "right"   # required for SFT loss masking


# ══════════════════════════════════════════════════════════════════════════
# 2. FORMAT DATA  →  chat template
# ══════════════════════════════════════════════════════════════════════════

print("[2] Formatting datasets with chat template …")

def apply_chat_template(sample: dict) -> dict:
    """
    Convert messages list → single string using the model's native chat template.
    SFTTrainer will tokenize this string; the collator masks everything
    except the assistant turn so loss is only computed on completions.
    """
    text = tokenizer.apply_chat_template(
        sample["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}

data_root = Path(CFG.data_root)
train_ds  = load_from_disk(str(data_root / "processed" / "train"))
val_ds    = load_from_disk(str(data_root / "processed" / "val"))

train_ds = train_ds.map(apply_chat_template, num_proc=4, remove_columns=train_ds.column_names)
val_ds   = val_ds.map(apply_chat_template,   num_proc=4, remove_columns=val_ds.column_names)

# Sanity check — print one formatted example
print("\n── Sample formatted input (first 400 chars) ──")
print(train_ds[0]["text"][:400])
print("…\n")


# ══════════════════════════════════════════════════════════════════════════
# 3. MODEL  —  4-bit NF4 quantization
# ══════════════════════════════════════════════════════════════════════════

print("[3] Loading model in 4-bit …")

bnb_config = BitsAndBytesConfig(
    load_in_4bit              = True,
    bnb_4bit_quant_type       = "nf4",           # NormalFloat4 — best for LLMs
    bnb_4bit_compute_dtype    = torch.bfloat16,  # compute in BF16
    bnb_4bit_use_double_quant = True,            # nested quantization → saves ~0.4 GB
)

model = AutoModelForCausalLM.from_pretrained(
    CFG.model_id,
    quantization_config  = bnb_config,
    device_map           = "auto",
    torch_dtype          = torch.bfloat16,
    attn_implementation  = "flash_attention_2",  # 2× faster; falls back gracefully
    trust_remote_code    = True,
)

# Required before adding LoRA to a quantized model
model = prepare_model_for_kbit_training(model)
model.config.use_cache = False   # disable KV-cache during training


# ══════════════════════════════════════════════════════════════════════════
# 4. LORA
# ══════════════════════════════════════════════════════════════════════════

print("[4] Attaching LoRA adapters …")

lora_config = LoraConfig(
    r               = CFG.lora_r,
    lora_alpha      = CFG.lora_alpha,
    lora_dropout    = CFG.lora_dropout,
    target_modules  = CFG.lora_target_modules,
    task_type       = TaskType.CAUSAL_LM,
    bias            = "none",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Expected output:
# trainable params: 41,943,040 || all params: 8,072,179,712 || trainable%: 0.52%


# ══════════════════════════════════════════════════════════════════════════
# 5. COMPLETION-ONLY LOSS MASKING
# ══════════════════════════════════════════════════════════════════════════
# Mask system + user tokens from loss so we ONLY train on assistant responses.
# This is critical — without it the model wastes capacity memorising prompts.

# LLaMA-3 assistant turn header (adjust if using Mistral or Gemma)
response_template = "<|start_header_id|>assistant<|end_header_id|>"
# Mistral: "[/INST]"
# Gemma:   "<start_of_turn>model\n"

collator = DataCollatorForCompletionOnlyLM(
    response_template = response_template,
    tokenizer         = tokenizer,
    mlm               = False,
)


# ══════════════════════════════════════════════════════════════════════════
# 6. TRAINING ARGUMENTS
# ══════════════════════════════════════════════════════════════════════════

print("[5] Building trainer …")

training_args = SFTConfig(
    output_dir                  = CFG.output_dir,
    num_train_epochs            = CFG.num_epochs,
    per_device_train_batch_size = CFG.per_device_train_batch_size,
    per_device_eval_batch_size  = CFG.per_device_train_batch_size,
    gradient_accumulation_steps = CFG.gradient_accumulation_steps,
    gradient_checkpointing      = True,   # halves VRAM at ~20% speed cost
    learning_rate               = CFG.learning_rate,
    lr_scheduler_type           = CFG.lr_scheduler_type,
    warmup_ratio                = CFG.warmup_ratio,
    weight_decay                = CFG.weight_decay,
    bf16                        = CFG.bf16,
    fp16                        = CFG.fp16,
    max_seq_length              = CFG.max_seq_length,
    dataset_text_field          = "text",
    logging_steps               = CFG.logging_steps,
    eval_strategy               = "steps",
    eval_steps                  = CFG.eval_steps,
    save_strategy               = "steps",
    save_steps                  = CFG.save_steps,
    save_total_limit            = CFG.save_total_limit,
    load_best_model_at_end      = True,
    metric_for_best_model       = "eval_loss",
    greater_is_better           = False,
    report_to                   = "wandb",
    run_name                    = wandb.run.name,
    seed                        = CFG.seed,
    dataloader_num_workers      = 4,
    packing                     = False,  # keep False — packing breaks completion masking
)


# ══════════════════════════════════════════════════════════════════════════
# 7. TRAINER
# ══════════════════════════════════════════════════════════════════════════

trainer = SFTTrainer(
    model           = model,
    tokenizer       = tokenizer,
    args            = training_args,
    train_dataset   = train_ds,
    eval_dataset    = val_ds,
    data_collator   = collator,
)


# ══════════════════════════════════════════════════════════════════════════
# 8. TRAIN
# ══════════════════════════════════════════════════════════════════════════

print("\n[6] Starting training …")
print(f"    Model        : {CFG.model_id}")
print(f"    Train samples: {len(train_ds):,}")
print(f"    Val samples  : {len(val_ds):,}")
print(f"    LoRA rank    : {CFG.lora_r}  alpha: {CFG.lora_alpha}")
print(f"    Eff. batch   : {CFG.per_device_train_batch_size * CFG.gradient_accumulation_steps}")
print(f"    Steps/epoch  : {math.ceil(len(train_ds) / (CFG.per_device_train_batch_size * CFG.gradient_accumulation_steps))}")
print()

trainer.train()


# ══════════════════════════════════════════════════════════════════════════
# 9. SAVE ADAPTER  (tiny: ~120 MB for r=16 with MLP targets)
# ══════════════════════════════════════════════════════════════════════════

adapter_path = Path(CFG.output_dir) / "adapter"
print(f"\n[7] Saving LoRA adapter → {adapter_path}")
model.save_pretrained(str(adapter_path))
tokenizer.save_pretrained(str(adapter_path))

# Log adapter size to W&B
import os
adapter_size_mb = sum(
    f.stat().st_size for f in adapter_path.rglob("*") if f.is_file()
) / 1e6
wandb.log({"adapter_size_mb": round(adapter_size_mb, 1)})
print(f"    Adapter size : {adapter_size_mb:.1f} MB")

wandb.finish()
print("\n✓ Training complete.")
