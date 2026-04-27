"""
MedBridge — Dataset Cleaner
=============================
Drops the 55 flagged rows identified by 02_validate_data.py:
  - AI disclaimers in output  (11 rows)
  - Excessive questions        (44 rows)

Run this ONCE after validation, before training.

Usage:
    python 03_clean_data.py
"""

import re, json
from pathlib import Path
from datasets import load_from_disk, DatasetDict
import shutil

DATA = Path(__file__).parent.parent / "data"

AI_DISCLAIMER_PHRASES = [
    "As an AI",
    "I cannot",
    "I'm sorry, I",
    "I don't have access",
    "As a language model",
    "I'm not able to provide",
    "I am not a doctor",
    "I am an AI",
]

def is_clean(sample: dict) -> bool:
    msgs = sample.get("messages", [])
    if len(msgs) < 3:
        return False

    output = msgs[2]["content"]

    # Drop AI disclaimers
    for phrase in AI_DISCLAIMER_PHRASES:
        if phrase in output[:120]:
            return False

    # Drop question dumps (>6 question marks)
    if output.count("?") > 6:
        return False

    return True

def clean_split(name: str, path: Path) -> dict:
    ds = load_from_disk(str(path))
    before = len(ds)
    ds = ds.filter(is_clean, num_proc=4)
    after = len(ds)
    dropped = before - after
    print(f"  {name:<8} {before:>6,} → {after:>6,}  (dropped {dropped})")
    temp_path = str(path) + "_clean"
    ds.save_to_disk(temp_path)
    shutil.rmtree(path)
    shutil.move(temp_path, path)
    return {"before": before, "after": after, "dropped": dropped}

if __name__ == "__main__":
    print("Cleaning splits …\n")
    print(f"  {'split':<8} {'before':>6}   {'after':>6}   dropped")
    print(f"  {'-'*42}")

    results = {}
    for split, path in [
        ("train", DATA / "processed" / "train"),
        ("val",   DATA / "processed" / "val"),
        ("test",  DATA / "eval" / "test"),
    ]:
        results[split] = clean_split(split, path)

    total_dropped = sum(r["dropped"] for r in results.values())
    total_after   = sum(r["after"]   for r in results.values())
    print(f"\n  Total dropped : {total_dropped}")
    print(f"  Total remaining : {total_after:,}")
    print("\n✓ Clean splits saved. Ready for training.")

    # Update stats.json
    stats_path = DATA / "stats.json"
    if stats_path.exists():
        with open(stats_path) as f:
            stats = json.load(f)
        for split, r in results.items():
            if split in stats:
                stats[split]["n"] = r["after"]
                stats[split]["dropped_by_cleaner"] = r["dropped"]
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"  stats.json updated → {stats_path}")
