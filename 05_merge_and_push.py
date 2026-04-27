"""
MedBridge — Merge LoRA Adapter into Base Model
================================================
Run this AFTER training to produce a standalone merged model.

Why merge?
- Adapter-only inference requires loading base model + adapter separately
- Merged model is a single HuggingFace model — works with vLLM, TGI, Ollama
- Merged model can be pushed directly to HuggingFace Hub

Output size: ~16 GB (BF16) or ~8 GB (FP16)

Usage:
    python 05_merge_and_push.py
    python 05_merge_and_push.py --push --hf-repo YOUR_USERNAME/medbridge-llm-8b
"""

import argparse, torch
from pathlib import Path
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

OUTPUT_DIR = Path("outputs/medbridge-v1")
ADAPTER    = OUTPUT_DIR / "adapter"
MERGED     = OUTPUT_DIR / "merged"


def merge(adapter_path: Path, merged_path: Path):
    print(f"\n[1] Loading adapter config from {adapter_path} …")
    config   = PeftConfig.from_pretrained(str(adapter_path))
    base_id  = config.base_model_name_or_path
    print(f"    Base model : {base_id}")

    print("[2] Loading base model in BF16 on CPU (safer for merge) …")
    base = AutoModelForCausalLM.from_pretrained(
        base_id,
        torch_dtype   = torch.bfloat16,
        device_map    = "cpu",
        trust_remote_code = True,
    )

    print("[3] Loading LoRA adapter …")
    model = PeftModel.from_pretrained(base, str(adapter_path))

    print("[4] Merging adapter weights into base (merge_and_unload) …")
    merged_model = model.merge_and_unload()

    print(f"[5] Saving merged model → {merged_path}")
    merged_path.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(str(merged_path), safe_serialization=True)

    print("[6] Saving tokenizer …")
    tok = AutoTokenizer.from_pretrained(str(adapter_path))
    tok.save_pretrained(str(merged_path))

    size_gb = sum(
        f.stat().st_size for f in merged_path.rglob("*.safetensors")
    ) / 1e9
    print(f"\n✓ Merged model saved  ({size_gb:.1f} GB)")
    return merged_path


def push_to_hub(merged_path: Path, repo_id: str):
    from huggingface_hub import HfApi
    print(f"\n[7] Pushing to HuggingFace Hub → {repo_id} …")
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(
        folder_path = str(merged_path),
        repo_id     = repo_id,
        repo_type   = "model",
        commit_message = "Add MedBridge fine-tuned model (QLoRA on AlpaCare-52k)",
    )
    print(f"✓ Pushed → https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter",  default=str(ADAPTER),  help="Path to LoRA adapter")
    parser.add_argument("--merged",   default=str(MERGED),   help="Where to save merged model")
    parser.add_argument("--push",     action="store_true",    help="Push to HuggingFace Hub")
    parser.add_argument("--hf-repo",  default=None,           help="HF repo id, e.g. username/medbridge-llm-8b")
    args = parser.parse_args()

    merged_path = merge(Path(args.adapter), Path(args.merged))

    if args.push:
        if not args.hf_repo:
            raise ValueError("--hf-repo is required when using --push")
        push_to_hub(merged_path, args.hf_repo)
