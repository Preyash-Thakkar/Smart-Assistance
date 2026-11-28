"""Staged hyperparameter/encoder tuning for the SetFit intent classifier.

Selection uses the grouped **validation** split only; the 31-batch test split is
never touched until the final retrain.  Each run writes a separate model under
``models/experiments/<name>`` and a validation report under
``evaluation/tuning/<name>_val.json``.

Usage::

    python training/tune.py --set minilm
    python training/tune.py --set encoders
    python training/tune.py --only it8_ep1_h16 mpnet_it8_ep1_h16
    python training/tune.py --list

Selection on a custom (e.g. ambient-inclusive) dev split::

    python training/tune.py --only it32_ep1_h16 \
        --train /tmp/opencode/train_tune.jsonl \
        --val /tmp/opencode/val_tune.jsonl \
        --tag dev --results evaluation/tuning/results_dev.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

TRAIN = os.path.join("data", "processed", "train.jsonl")
VAL = os.path.join("data", "processed", "val.jsonl")
EXP_DIR = os.path.join("models", "experiments")
TUNE_DIR = os.path.join("evaluation", "tuning")
RESULTS = os.path.join(TUNE_DIR, "results.json")

BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

MINILM_CONFIGS = [
    {"name": "it4_ep1_h16", "iterations": 4, "epochs": 1, "head_epochs": 16},
    {"name": "it8_ep1_h16", "iterations": 8, "epochs": 1, "head_epochs": 16},
    {"name": "it16_ep1_h16", "iterations": 16, "epochs": 1, "head_epochs": 16},
    {"name": "it16_ep1_h8", "iterations": 16, "epochs": 1, "head_epochs": 8},
    {"name": "it16_ep1_h32", "iterations": 16, "epochs": 1, "head_epochs": 32},
    {"name": "it16_ep2_h16", "iterations": 16, "epochs": 2, "head_epochs": 16},
    {"name": "it16_ep1_h16_bs32", "iterations": 16, "epochs": 1, "head_epochs": 16, "batch_size": 32},
    {"name": "it32_ep1_h16", "iterations": 32, "epochs": 1, "head_epochs": 16},
    {"name": "it32_ep2_h16", "iterations": 32, "epochs": 2, "head_epochs": 16},
    {"name": "it64_ep1_h16", "iterations": 64, "epochs": 1, "head_epochs": 16},
    {"name": "it8_ep2_h16", "iterations": 8, "epochs": 2, "head_epochs": 16},
    {"name": "it8_ep1_h8", "iterations": 8, "epochs": 1, "head_epochs": 8},
    {"name": "it8_ep1_h4", "iterations": 8, "epochs": 1, "head_epochs": 4},
    {"name": "it8_ep1_h32", "iterations": 8, "epochs": 1, "head_epochs": 32},
    {"name": "it8_ep0_h32", "iterations": 8, "epochs": 0, "head_epochs": 32},
    {"name": "it8_ep1_h16_bs32", "iterations": 8, "epochs": 1, "head_epochs": 16, "batch_size": 32},
]

ENCODER_CONFIGS = [
    {
        "name": "mpnet_it32_ep1_h16",
        "base_model": "sentence-transformers/all-mpnet-base-v2",
        "iterations": 32,
        "epochs": 1,
        "head_epochs": 16,
    },
    {
        "name": "bge_it32_ep1_h16",
        "base_model": "BAAI/bge-small-en-v1.5",
        "iterations": 32,
        "epochs": 1,
        "head_epochs": 16,
    },
    {
        "name": "minilm12_it32_ep1_h16",
        "base_model": "sentence-transformers/all-MiniLM-L12-v2",
        "iterations": 32,
        "epochs": 1,
        "head_epochs": 16,
    },
    {
        "name": "minilm12_it64_ep1_h16",
        "base_model": "sentence-transformers/all-MiniLM-L12-v2",
        "iterations": 64,
        "epochs": 1,
        "head_epochs": 16,
    },
]

SETS = {
    "minilm": MINILM_CONFIGS,
    "encoders": ENCODER_CONFIGS,
    "all": MINILM_CONFIGS + ENCODER_CONFIGS,
}


def load_results(path: str = RESULTS) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_results(results: dict, path: str = RESULTS) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, sort_keys=True)
        handle.write("\n")


def run_config(config: dict, train_path: str, val_path: str, tag: str = "") -> dict:
    name = config["name"] + (f"_{tag}" if tag else "")
    model_dir = os.path.join(EXP_DIR, name)
    train_log = os.path.join(TUNE_DIR, f"{name}_train.log")
    val_report = os.path.join(TUNE_DIR, f"{name}_val.json")
    val_matrix = os.path.join(TUNE_DIR, f"{name}_val_matrix.csv")

    command = [
        sys.executable,
        "-u",
        os.path.join("training", "train_classifier.py"),
        "--train",
        train_path,
        "--val",
        val_path,
        "--output",
        model_dir,
        "--base-model",
        config.get("base_model", BASE_MODEL),
        "--iterations",
        str(config.get("iterations", 4)),
        "--epochs",
        str(config.get("epochs", 1)),
        "--head-epochs",
        str(config.get("head_epochs", 16)),
        "--batch-size",
        str(config.get("batch_size", 16)),
        "--seed",
        str(config.get("seed", 42)),
    ]
    print(f"\n=== {name} ===", flush=True)
    print(" ".join(command), flush=True)
    with open(train_log, "w", encoding="utf-8") as handle:
        completed = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT)
    if completed.returncode != 0:
        return {"name": name, "status": "train_failed", "returncode": completed.returncode}

    eval_command = [
        sys.executable,
        "-u",
        os.path.join("evaluation", "evaluate.py"),
        "--split",
        "val",
        "--data",
        val_path,
        "--train",
        train_path,
        "--model-dir",
        model_dir,
        "--report-out",
        val_report,
        "--matrix-out",
        val_matrix,
    ]
    eval_log = os.path.join(TUNE_DIR, f"{name}_eval.log")
    with open(eval_log, "w", encoding="utf-8") as handle:
        completed = subprocess.run(eval_command, stdout=handle, stderr=subprocess.STDOUT)
    if completed.returncode != 0:
        return {"name": name, "status": "eval_failed", "returncode": completed.returncode}

    with open(val_report, "r", encoding="utf-8") as handle:
        report = json.load(handle)
    result = {
        "name": name,
        "status": "ok",
        "config": config,
        "val_accuracy": report["accuracy"],
        "val_macro_f1": report["macro_f1"],
        "val_weighted_f1": report["weighted_f1"],
        "val_rows": report["rows"],
        "model_dir": model_dir,
    }
    print(
        f"{name:24s} val acc={result['val_accuracy']*100:5.2f}%  "
        f"macro={result['val_macro_f1']*100:5.2f}%  "
        f"weighted={result['val_weighted_f1']*100:5.2f}%",
        flush=True,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", choices=sorted(SETS), default="minilm")
    parser.add_argument("--train", default=TRAIN)
    parser.add_argument("--val", default=VAL)
    parser.add_argument("--only", nargs="*", default=None, help="Run only these config names")
    parser.add_argument(
        "--tag",
        default="",
        help="Suffix for run names/reports, e.g. dev (keeps official runs intact)",
    )
    parser.add_argument(
        "--results",
        default=RESULTS,
        help="Results json path (use a separate file for non-official splits)",
    )
    parser.add_argument("--force", action="store_true", help="Re-run configs that already have results")
    parser.add_argument("--list", action="store_true", help="List configs and exit")
    args = parser.parse_args()

    configs = SETS[args.set]
    if args.only:
        wanted = set(args.only)
        configs = [c for c in SETS["all"] if c["name"] in wanted]
        missing = wanted - {c["name"] for c in configs}
        if missing:
            raise SystemExit(f"Unknown config(s): {sorted(missing)}")
    if args.list:
        for config in SETS["all"]:
            print(config["name"], config)
        return

    os.makedirs(EXP_DIR, exist_ok=True)
    os.makedirs(TUNE_DIR, exist_ok=True)
    results = load_results(args.results)

    for config in configs:
        name = config["name"] + (f"_{args.tag}" if args.tag else "")
        if name in results and results[name].get("status") == "ok" and not args.force:
            print(f"skip {name} (already has results; use --force to rerun)", flush=True)
            continue
        result = run_config(config, args.train, args.val, args.tag)
        results[name] = result
        save_results(results, args.results)

    ok = [r for r in results.values() if r.get("status") == "ok"]
    if ok:
        print("\nValidation ranking (macro-F1):", flush=True)
        for result in sorted(ok, key=lambda r: -r["val_macro_f1"]):
            print(
                f"  {result['name']:24s} "
                f"acc={result['val_accuracy']*100:5.2f}%  "
                f"macro={result['val_macro_f1']*100:5.2f}%  "
                f"weighted={result['val_weighted_f1']*100:5.2f}%",
                flush=True,
            )


if __name__ == "__main__":
    main()
