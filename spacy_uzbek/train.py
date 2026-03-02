#!/usr/bin/env python3
"""
End-to-end spaCy training pipeline for Uzbek.

This script orchestrates:
  1. Registering the custom Uzbek language class
  2. Converting CoNLL-U data to spaCy DocBin format
  3. (Optional) Converting FastText vectors to spaCy format
  4. Training the spaCy pipeline (tagger + morphologizer + parser)
  5. Evaluating on the test set

Usage:
    # Transformer-based (GPU recommended):
    python spacy_uzbek/train.py --config transformer --data combined --gpu 0

    # Static vectors baseline (CPU):
    python spacy_uzbek/train.py --config static --data combined

    # Single treebank:
    python spacy_uzbek/train.py --config transformer --data uzudt --gpu 0

See README.md § spaCy Experiments for detailed instructions.
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def register_uzbek():
    """Register the Uzbek language class with spaCy."""
    sys.path.insert(0, str(PROJECT_ROOT))
    from spacy_uzbek.lang.uz import Uzbek  # noqa: F401
    print("[1/5] Uzbek language class registered with spaCy.")


def convert_data(data_setting: str):
    """Convert CoNLL-U files to spaCy DocBin format."""
    print(f"\n[2/5] Converting CoNLL-U data to spaCy format (setting: {data_setting})...")

    from spacy_uzbek.convert_conllu import conllu_to_docbin

    data_dir = PROJECT_ROOT / "spacy_uzbek" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    if data_setting == "uzudt":
        splits = {
            "train": "data/pos/uz_uzudt.train.in.conllu",
            "dev":   "data/pos/uz_uzudt.dev.in.conllu",
            "test":  "data/pos/uz_uzudt.test.in.conllu",
        }
        prefix = "uz_uzudt"
    elif data_setting == "ut":
        splits = {
            "train": "data/pos/uz_ut-ud-train.in.conllu",
            "dev":   "data/pos/uz_ut-ud-dev.in.conllu",
            "test":  "data/pos/uz_ut-ud-test.in.conllu",
        }
        prefix = "uz_ut"
    elif data_setting == "combined":
        splits = {
            "train": "data/pos/merged/uz_combined.train.in.conllu",
            "dev":   "data/pos/merged/uz_combined.dev.in.conllu",
            "test":  "data/pos/merged/uz_combined.test.in.conllu",
        }
        prefix = "uz_combined"
    else:
        raise ValueError(f"Unknown data setting: {data_setting}")

    paths = {}
    for split_name, conllu_rel in splits.items():
        conllu_path = PROJECT_ROOT / conllu_rel
        spacy_path = data_dir / f"{prefix}.{split_name}.spacy"
        if not conllu_path.exists():
            print(f"  WARNING: {conllu_path} not found, skipping.")
            continue
        conllu_to_docbin(str(conllu_path), str(spacy_path))
        paths[split_name] = str(spacy_path)

    return paths


def convert_vectors():
    """
    Convert FastText .vec file to spaCy vectors directory format.

    This creates a spaCy-compatible vectors directory at
    wordvec/uz/spacy_vectors/ that can be referenced in the static config.
    """
    vec_file = PROJECT_ROOT / "wordvec" / "uz" / "fasttext" / "cc.uz.300.vec"
    output_dir = PROJECT_ROOT / "wordvec" / "uz" / "spacy_vectors"

    if output_dir.exists() and (output_dir / "vectors").exists():
        print(f"\n[3/5] spaCy vectors already exist at {output_dir}, skipping.")
        return str(output_dir)

    if not vec_file.exists():
        print(f"\n[3/5] WARNING: FastText vectors not found at {vec_file}.")
        print("  Static vector config will not work without vectors.")
        print("  Download: https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.uz.300.vec.gz")
        return None

    print(f"\n[3/5] Converting FastText vectors to spaCy format...")
    print(f"  This may take a few minutes for large vector files...")

    # Use spaCy CLI to init vectors
    cmd = [
        sys.executable, "-m", "spacy", "init", "vectors",
        "uz",
        str(vec_file),
        str(output_dir),
        "--truncate", "50000",  # Keep top 50k vectors to save memory
        "--name", "uz_fasttext_vectors",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR converting vectors: {result.stderr}")
        return None
    print(f"  Vectors saved to {output_dir}")
    return str(output_dir)


def train_model(config_type: str, data_paths: dict, vectors_path: str = None,
                gpu_id: int = -1, output_dir: str = None):
    """
    Run spaCy training.

    Parameters
    ----------
    config_type : str
        "transformer" or "static"
    data_paths : dict
        {"train": path, "dev": path, "test": path}
    vectors_path : str or None
        Path to spaCy vectors directory (for static config)
    gpu_id : int
        GPU device ID (-1 for CPU)
    output_dir : str
        Model output directory
    """
    config_dir = PROJECT_ROOT / "spacy_uzbek" / "configs"
    if config_type == "transformer":
        config_file = config_dir / "config_transformer.cfg"
    else:
        config_file = config_dir / "config_static.cfg"

    if output_dir is None:
        output_dir = str(PROJECT_ROOT / "saved_models" / "spacy" / config_type)

    print(f"\n[4/5] Training spaCy model...")
    print(f"  Config: {config_file}")
    print(f"  Output: {output_dir}")
    print(f"  GPU:    {gpu_id if gpu_id >= 0 else 'CPU'}")

    cmd = [
        sys.executable, "-m", "spacy", "train",
        str(config_file),
        "--output", output_dir,
        "--paths.train", data_paths["train"],
        "--paths.dev", data_paths["dev"],
    ]

    if gpu_id >= 0:
        cmd.extend(["--gpu-id", str(gpu_id)])

    if vectors_path and config_type == "static":
        cmd.extend(["--paths.vectors", vectors_path])

    print(f"\n  Command: {' '.join(cmd)}\n")
    print("=" * 70)

    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        print(f"\n  Training failed with exit code {result.returncode}")
        sys.exit(1)

    print(f"\n  Model saved to {output_dir}")
    return output_dir


def evaluate_model(model_dir: str, test_path: str, gpu_id: int = -1):
    """Evaluate a trained spaCy model on the test set."""
    model_best = Path(model_dir) / "model-best"
    if not model_best.exists():
        print(f"\n[5/5] WARNING: model-best not found at {model_best}")
        return

    print(f"\n[5/5] Evaluating model on test set...")
    print(f"  Model: {model_best}")
    print(f"  Test:  {test_path}")

    cmd = [
        sys.executable, "-m", "spacy", "evaluate",
        str(model_best),
        test_path,
        "--output", str(Path(model_dir) / "test_metrics.json"),
    ]

    if gpu_id >= 0:
        cmd.extend(["--gpu-id", str(gpu_id)])

    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        print(f"  Evaluation failed with exit code {result.returncode}")
    else:
        print(f"\n  Metrics saved to {Path(model_dir) / 'test_metrics.json'}")


def main():
    parser = argparse.ArgumentParser(
        description="Train a spaCy tagger + parser pipeline for Uzbek"
    )
    parser.add_argument(
        "--config", "-c", type=str, choices=["transformer", "static"],
        default="transformer",
        help="Config type: 'transformer' (TahrirchiBERT) or 'static' (FastText)"
    )
    parser.add_argument(
        "--data", "-d", type=str, choices=["uzudt", "ut", "combined"],
        default="combined",
        help="Data setting: 'uzudt', 'ut', or 'combined' (merged)"
    )
    parser.add_argument(
        "--gpu", "-g", type=int, default=-1,
        help="GPU device ID (-1 for CPU)"
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output directory for trained model"
    )
    parser.add_argument(
        "--skip-convert", action="store_true",
        help="Skip data conversion (use existing .spacy files)"
    )
    parser.add_argument(
        "--eval-only", action="store_true",
        help="Only evaluate an existing model (requires --output)"
    )
    args = parser.parse_args()

    # Step 1: Register language
    register_uzbek()

    # Step 2: Convert data
    if args.skip_convert or args.eval_only:
        data_dir = PROJECT_ROOT / "spacy_uzbek" / "data"
        prefix_map = {"uzudt": "uz_uzudt", "ut": "uz_ut", "combined": "uz_combined"}
        prefix = prefix_map[args.data]
        data_paths = {
            "train": str(data_dir / f"{prefix}.train.spacy"),
            "dev":   str(data_dir / f"{prefix}.dev.spacy"),
            "test":  str(data_dir / f"{prefix}.test.spacy"),
        }
    else:
        data_paths = convert_data(args.data)

    # Step 3: Convert vectors (if static)
    vectors_path = None
    if args.config == "static":
        vectors_path = convert_vectors()

    # Step 4: Train (unless eval-only)
    output_dir = args.output
    if output_dir is None:
        output_dir = str(
            PROJECT_ROOT / "saved_models" / "spacy"
            / f"{args.config}_{args.data}"
        )

    if not args.eval_only:
        train_model(
            config_type=args.config,
            data_paths=data_paths,
            vectors_path=vectors_path,
            gpu_id=args.gpu,
            output_dir=output_dir,
        )

    # Step 5: Evaluate
    if "test" in data_paths:
        evaluate_model(output_dir, data_paths["test"], gpu_id=args.gpu)


if __name__ == "__main__":
    main()
