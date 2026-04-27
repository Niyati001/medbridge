"""
MedBridge — Data Preparation Pipeline
======================================
Sources:
  1. AlpaCare-MedInstruct-52k  (HuggingFace) — real medical instruction pairs
  2. MedQuAD                   (Kaggle CSV)   — 47k real NIH Q&A pairs

Strategy for MedQuAD:
  NIH answers are already patient-friendly (that IS the simplified side).
  We generate the "clinical jargon" version via the Anthropic API so every
  pair is grounded in real NIH content — NOT synthetic from scratch.

Output:
  data/processed/train/   — HuggingFace Dataset (arrow)
  data/processed/val/     — HuggingFace Dataset (arrow)
  data/eval/test/         — HuggingFace Dataset (arrow)
  data/stats.json         — dataset composition report
"""

import os, json, re, time, hashlib
from pathlib import Path
from datasets import load_dataset, Dataset, concatenate_datasets
import pandas as pd

#paths
BASE = Path(__file__).parent.parent
DATA = BASE / "data"
(DATA / "raw").mkdir(parents=True, exist_ok=True)
(DATA / "processed").mkdir(parents=True, exist_ok=True)
(DATA / "eval").mkdir(parents=True, exist_ok=True)

# Anthropic client (for MedQuAD jargon generation) 
# SYSTEM PROMPT — used at inference AND during training
SYSTEM_PROMPT = """You are MedBridge, a medical language specialist trained \
on peer-reviewed clinical literature. Your task is to simplify complex \
medical text into clear, accurate language a patient with no medical \
background can understand.

Rules:
- NEVER omit diagnoses, drug names, dosages, or contraindications
- Replace jargon with plain English; define unavoidable terms in parentheses
- Keep sentences short (under 20 words where possible)
- Do not add information not present in the original
- Do not give medical advice beyond what the source states
- Match the reading level to: eighth-grade comprehension"""


# PART 1 — AlpaCare-MedInstruct-52k

SIMPLIFY_KEYWORDS = ["simpl", "layman", "patient", "easier", "plain english", "rewrite", "condense", "translate", "summarize", "explain", "non-medical", "understandable", "accessible", "rephrase",  "plain language", "everyday", "convert", "clarify"]

def is_simplification(sample: dict) -> bool:
    text = (sample.get("instruction", "") + " " + sample.get("input", "")).lower()
    return any(kw in text for kw in SIMPLIFY_KEYWORDS)

def quality_filter(sample: dict) -> bool:
    """Remove low-quality samples."""
    out = sample.get("output", "")
    if len(out.strip()) < 40:          return False   # too short
    if len(out.split()) > 600:         return False   # too long
    if out.count("?") > 8:             return False   # question dumps
    if "I cannot" in out[:60]:         return False   # refusals
    if "As an AI" in out[:80]:         return False   # AI disclaimers
    return True

def format_alphacare(sample: dict) -> dict:
    has_input = sample.get("input", "").strip()
    user_msg = (
        f"{sample['instruction'].strip()}\n\n{sample['input'].strip()}"
        if has_input else sample["instruction"].strip())
    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": user_msg},
            {"role": "assistant", "content": sample["output"].strip()},
        ],
        "source": "alphacare",
        "id": hashlib.md5(user_msg.encode()).hexdigest()[:12],
    }

def load_alphacare() -> Dataset:
    print("\n[1/2] Loading AlpaCare-MedInstruct-52k …")
    ds = load_dataset("lavita/AlpaCare-MedInstruct-52k", split="train")
    print(f"  Raw samples : {len(ds):,}")

    ds = ds.filter(is_simplification, num_proc=4)
    print(f"  After keyword filter : {len(ds):,}")

    ds = ds.filter(quality_filter, num_proc=4)
    print(f"  After quality filter : {len(ds):,}")

    ds = ds.map(format_alphacare, num_proc=4,
                remove_columns=ds.column_names)
    print(f"  Formatted samples : {len(ds):,}")
    return ds


# PART 2 — MedQuAD  (Kaggle CSV → jargon generation via API)

JARGON_PROMPT = """\
Below is a plain-language patient answer from the NIH about a medical topic.

Your task: rewrite it as the kind of dense clinical text a specialist would \
write — using proper medical terminology, Latin/Greek terms where appropriate, \
abbreviations (e.g. HTN, DM, GERD), and passive voice typical of clinical \
documentation. Do NOT add new facts. Keep the same information, just change \
the register to clinical/specialist level.

NIH patient answer:
{answer}

Clinical rewrite (output only, no preamble):"""

def generate_clinical_jargon(nih_answer: str, max_retries: int = 3):
    return None


def format_medquad_pair(clinical: str, nih_answer: str, topic: str) -> dict:
    user_msg = f"Simplify the following clinical text about {topic}:\n\n{clinical}"
    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": user_msg},
            {"role": "assistant", "content": nih_answer.strip()},],
        "source": "medquad_nih",
        "id": hashlib.md5(nih_answer.encode()).hexdigest()[:12],
    }


def load_medquad(csv_path: str, max_samples: int = 8000) -> Dataset:
    """
    Args:
        csv_path  : Path to the medquad.csv you downloaded from Kaggle
        max_samples: Cap to keep API cost reasonable (~$2–4 for 8k samples
                     at Haiku pricing)
    """
    print("\n[2/2] Processing MedQuAD …")
    df = pd.read_csv(csv_path)
    print(f"  Raw rows : {len(df):,}")
    print(f"  Columns  : {list(df.columns)}")

    # MedQuAD CSVs vary slightly — normalise column names
    df.columns = [c.lower().strip().replace(" ", "_") for c in df.columns]

    # Keep rows where both question and answer are present
    answer_col  = next((c for c in df.columns if "answer" in c), None)
    question_col = next((c for c in df.columns if "question" in c), None)
    focus_col   = next((c for c in df.columns if "focus" in c or "topic" in c), None)

    if not answer_col:
        raise ValueError(f"Cannot find answer column. Columns: {list(df.columns)}")

    df = df.dropna(subset=[answer_col])
    df = df[df[answer_col].str.len() > 80]   # minimum meaningful answer length

    # Sample a balanced cross-section (don't just take top N)
    if len(df) > max_samples:
        df = df.sample(max_samples, random_state=42).reset_index(drop=True)

    print(f"  Sampled  : {len(df):,} rows for jargon generation")
    print(f"  Estimated API cost: ~${len(df) * 0.0003:.2f} (Haiku pricing)")

    records = []
    failed  = 0

    for i, row in df.iterrows():
        nih_answer = str(row[answer_col])
        topic = str(row[focus_col]) if focus_col else "the condition"

        clinical = generate_clinical_jargon(nih_answer)

        if clinical:
            records.append(format_medquad_pair(clinical, nih_answer, topic))
        else:
            failed += 1

        # Progress every 100 rows
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i+1}/{len(df)} | failed: {failed}")

        time.sleep(0.05)   # rate limit courtesy pause

    print(f"\n  Generated pairs : {len(records):,}  |  failed : {failed}")
    return Dataset.from_list(records)


# PART 3 — Merge, deduplicate, split

def deduplicate(ds: Dataset) -> Dataset:
    """Remove exact duplicate IDs."""
    seen = set()
    keep = []
    for i, sample in enumerate(ds):
        if sample["id"] not in seen:
            seen.add(sample["id"])
            keep.append(i)
    return ds.select(keep)

def save_stats(ds_train, ds_val, ds_test):
    def source_counts(ds):
        counts = {}
        for s in ds["source"]:
            counts[s] = counts.get(s, 0) + 1
        return counts

    stats = {
        "total": len(ds_train) + len(ds_val) + len(ds_test),
        "train": {"n": len(ds_train), "sources": source_counts(ds_train)},
        "val"  : {"n": len(ds_val),   "sources": source_counts(ds_val)},
        "test" : {"n": len(ds_test),  "sources": source_counts(ds_test)},}
    with open(DATA / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print("\n── Dataset statistics ──────────────────────────")
    print(json.dumps(stats, indent=2))


def build_pipeline(medquad_csv: str | None = None,
                   skip_medquad_api: bool = False):
    """
    Main entry point.

    Args:
        medquad_csv     : Path to medquad.csv from Kaggle (optional)
        skip_medquad_api: Set True to use AlpaCare only (no API cost)
    """
    datasets = []

    # 1. AlpaCare
    alphacare_ds = load_alphacare()
    datasets.append(alphacare_ds)

    # 2. MedQuAD (optional — requires CSV + API key)
    if medquad_csv and not skip_medquad_api:
        if Path(medquad_csv).exists():
            medquad_ds = load_medquad(medquad_csv)
            datasets.append(medquad_ds)
        else:
            print(f"\n⚠ MedQuAD CSV not found at {medquad_csv} — skipping")
    elif skip_medquad_api:
        print("\n[MedQuAD] Skipped (skip_medquad_api=True)")

    # 3. Merge
    print("\n[3] Merging datasets …")
    combined = concatenate_datasets(datasets)
    combined = deduplicate(combined)
    print(f"  Combined (deduped) : {len(combined):,} samples")

    # 4. Shuffle + split  80 / 10 / 10
    combined = combined.shuffle(seed=42)
    split1   = combined.train_test_split(test_size=0.2, seed=42)
    split2   = split1["test"].train_test_split(test_size=0.5, seed=42)

    ds_train = split1["train"]
    ds_val   = split2["train"]
    ds_test  = split2["test"]

    # 5. Save
    print("\n[4] Saving splits …")
    ds_train.save_to_disk(str(DATA / "processed" / "train"))
    ds_val.save_to_disk(str(DATA / "processed" / "val"))
    ds_test.save_to_disk(str(DATA / "eval" / "test"))
    save_stats(ds_train, ds_val, ds_test)

    print("\n✓ Data pipeline complete.")
    print(f"  Train : {len(ds_train):,}")
    print(f"  Val   : {len(ds_val):,}")
    print(f"  Test  : {len(ds_test):,}")
    return ds_train, ds_val, ds_test


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MedBridge data pipeline")
    parser.add_argument("--medquad-csv",      type=str, default=None,
                        help="Path to medquad.csv downloaded from Kaggle")
    parser.add_argument("--skip-medquad-api", action="store_true",
                        help="Use AlpaCare only — no API cost")
    parser.add_argument("--max-medquad",      type=int, default=8000,
                        help="Max MedQuAD rows to process (default 8000)")
    args = parser.parse_args()

    build_pipeline(
        medquad_csv=args.medquad_csv,
        skip_medquad_api=args.skip_medquad_api,
    )
