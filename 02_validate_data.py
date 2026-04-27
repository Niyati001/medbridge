"""
MedBridge — Dataset Quality Validator
=======================================
Run this AFTER 01_prepare_data.py to audit your training data
before spending GPU hours on a bad dataset.

Checks:
  - Message format integrity
  - Length distribution
  - Source balance
  - Jargon density (input should be denser than output)
  - Duplicate rate
  - Samples that slipped through (AI disclaimers, refusals, etc.)
"""

import json, re
from pathlib import Path
from collections import Counter
from datasets import load_from_disk
import numpy as np

DATA = Path(__file__).parent.parent / "data"

# ── readability helpers ────────────────────────────────────────────────────
MEDICAL_JARGON = [
    r'\b(hypertension|HTN|myocardial|infarction|dyspnea|edema|'
    r'tachycardia|bradycardia|hepatic|renal|pulmonary|cardiac|'
    r'dyslipidemia|hyperlipidemia|atherosclerosis|ischemia|'
    r'hemorrhage|thrombosis|embolism|stenosis|fibrosis|necrosis|'
    r'carcinoma|lymphoma|leukocyte|erythrocyte|platelet|serum|'
    r'bilirubin|creatinine|hemoglobin|albumin|electrolyte)\b'
]
JARGON_RE = re.compile("|".join(MEDICAL_JARGON), re.IGNORECASE)

def jargon_density(text: str) -> float:
    words = text.split()
    if not words: return 0.0
    return len(JARGON_RE.findall(text)) / len(words)

def avg_words(texts):
    return np.mean([len(t.split()) for t in texts])

def syllables(word):
    word = word.lower()
    count = len(re.findall(r'[aeiou]+', word))
    return max(1, count)

def flesch_kincaid_grade(text):
    sentences = re.split(r'[.!?]+', text)
    sentences = [s for s in sentences if s.strip()]
    words = text.split()
    if not sentences or not words: return 0
    asl = len(words) / len(sentences)
    asw = sum(syllables(w) for w in words) / len(words)
    return 0.39 * asl + 11.8 * asw - 15.59


# ── validation ─────────────────────────────────────────────────────────────
def validate_split(split_name: str, path: Path):
    print(f"\n{'═'*55}")
    print(f"  Validating : {split_name}  ({path})")
    print(f"{'═'*55}")

    ds = load_from_disk(str(path))
    print(f"  Total samples : {len(ds):,}")

    issues = []
    user_texts, assistant_texts, sources = [], [], []

    for i, sample in enumerate(ds):
        msgs = sample.get("messages", [])

        # ── format check ──
        roles = [m["role"] for m in msgs]
        if roles != ["system", "user", "assistant"]:
            issues.append(f"Row {i}: bad role sequence {roles}")
            continue

        user_txt  = msgs[1]["content"]
        asst_txt  = msgs[2]["content"]
        src       = sample.get("source", "unknown")

        # ── content checks ──
        if len(asst_txt.strip()) < 30:
            issues.append(f"Row {i}: output too short ({len(asst_txt)} chars)")
        if any(phrase in asst_txt[:100] for phrase in
               ["As an AI", "I cannot", "I'm sorry, I", "I don't have access"]):
            issues.append(f"Row {i}: AI disclaimer in output")
        if asst_txt.count("?") > 6:
            issues.append(f"Row {i}: excessive questions in output")

        user_texts.append(user_txt)
        assistant_texts.append(asst_txt)
        sources.append(src)

    # ── aggregate stats ──
    print(f"\n  Source distribution:")
    for src, count in Counter(sources).most_common():
        pct = count / len(ds) * 100
        bar = "█" * int(pct / 2)
        print(f"    {src:<22} {count:>6,}  ({pct:>5.1f}%)  {bar}")

    print(f"\n  Length stats (words):")
    u_words = [len(t.split()) for t in user_texts]
    a_words = [len(t.split()) for t in assistant_texts]
    print(f"    User (input)   — mean:{np.mean(u_words):>6.0f}  "
          f"p5:{np.percentile(u_words,5):>4.0f}  p95:{np.percentile(u_words,95):>4.0f}")
    print(f"    Asst (output)  — mean:{np.mean(a_words):>6.0f}  "
          f"p5:{np.percentile(a_words,5):>4.0f}  p95:{np.percentile(a_words,95):>4.0f}")

    print(f"\n  Jargon density (higher = more clinical):")
    u_jargon = [jargon_density(t) for t in user_texts]
    a_jargon = [jargon_density(t) for t in assistant_texts]
    print(f"    User (input)  : {np.mean(u_jargon):.4f}")
    print(f"    Asst (output) : {np.mean(a_jargon):.4f}")
    ratio = np.mean(u_jargon) / (np.mean(a_jargon) + 1e-9)
    status = "✓ GOOD" if ratio >= 1.2 else "⚠ CHECK — output may be as dense as input"
    print(f"    Ratio (input/output): {ratio:.2f}  {status}")

    print(f"\n  Readability (Flesch-Kincaid grade level):")
    u_fk = [flesch_kincaid_grade(t) for t in user_texts[:500]]
    a_fk = [flesch_kincaid_grade(t) for t in assistant_texts[:500]]
    print(f"    User (input)  : grade {np.mean(u_fk):.1f}")
    print(f"    Asst (output) : grade {np.mean(a_fk):.1f}")
    target_ok = np.mean(a_fk) < np.mean(u_fk)
    print(f"    {'✓ Output is simpler than input' if target_ok else '⚠ Output not clearly simpler'}")

    print(f"\n  Duplicate ID check:")
    ids = [s.get("id", "") for s in ds]
    dupes = len(ids) - len(set(ids))
    print(f"    Duplicates : {dupes}  {'✓' if dupes == 0 else '⚠ Run dedup again'}")

    print(f"\n  Issues found : {len(issues)}")
    for issue in issues[:20]:   # show first 20
        print(f"    ⚠ {issue}")
    if len(issues) > 20:
        print(f"    … and {len(issues)-20} more")

    ok_rate = (len(ds) - len(issues)) / len(ds) * 100
    print(f"\n  Quality rate : {ok_rate:.1f}%  "
          f"{'✓ READY FOR TRAINING' if ok_rate > 97 else '⚠ REVIEW ISSUES BEFORE TRAINING'}")

    return {
        "split": split_name,
        "n": len(ds),
        "issues": len(issues),
        "quality_rate": round(ok_rate, 2),
        "jargon_ratio_input_vs_output": round(ratio, 3),
        "mean_grade_input": round(float(np.mean(u_fk)), 2),
        "mean_grade_output": round(float(np.mean(a_fk)), 2),
    }


if __name__ == "__main__":
    results = {}

    for split, path in [
        ("train", DATA / "processed" / "train"),
        ("val",   DATA / "processed" / "val"),
        ("test",  DATA / "eval" / "test"),
    ]:
        if path.exists():
            results[split] = validate_split(split, path)
        else:
            print(f"\n[skip] {split} not found at {path}")

    print(f"\n{'═'*55}")
    print("  VALIDATION SUMMARY")
    print(f"{'═'*55}")
    print(json.dumps(results, indent=2))

    report_path = DATA / "validation_report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Report saved → {report_path}")
