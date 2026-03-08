"""
mathvision.py  ─  Preprocessing script for multimodal math datasets.

Converts image+text math QA datasets (e.g. MathVista, GeoQA, MATH-Vision)
into the JSON format expected by CoLaR's QSADataModule:

  [
    {
      "question": "In the figure, ...",
      "answer": "15",
      "steps": ["step 1 ...", "step 2 ..."],
      "image_path": "images/train_001.png"   ← path relative to the dataset dir
    },
    ...
  ]

Usage
-----
  python data_preprocessing/mathvision.py \
      --source_dir /path/to/raw/mathvision \
      --output_dir /path/to/workspace/datasets/text_reasoning/mathvision

The script creates:
  <output_dir>/train.json
  <output_dir>/val.json
  <output_dir>/test.json
  <output_dir>/images/          ← images copied (or symlinked) here
"""

import argparse
import json
import random
import shutil
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Adapter functions for specific datasets
# Add your own adapter here following the same pattern.
# ─────────────────────────────────────────────────────────────────────────────

def load_mathvista(source_dir: Path):
    """
    MathVista (https://mathvista.github.io/).
    Expects the original testmini.json / test.json from the official release.
    """
    records = []
    for split_file in source_dir.glob("*.json"):
        with open(split_file) as f:
            raw = json.load(f)
        # MathVista is a dict: {id: {question, answer, image, solution, ...}}
        if isinstance(raw, dict):
            raw = list(raw.values())
        for item in raw:
            image_file = item.get("image", None)
            records.append({
                "question": item["question"],
                "answer": str(item.get("answer", item.get("gt_answer", ""))),
                # MathVista does not always provide step-by-step; use solution if present
                "steps": _split_solution(item.get("solution", item.get("rationale", ""))),
                "image_path": f"images/{image_file}" if image_file else None,
                "_raw_image_src": str(source_dir / image_file) if image_file else None,
            })
    return records


def load_geoqa(source_dir: Path):
    """
    GeoQA (https://github.com/chen-judge/GeoQA).
    Expects the official JSON splits and an 'images/' subfolder.
    """
    records = []
    for split_file in ["train.json", "val.json", "test.json"]:
        p = source_dir / split_file
        if not p.exists():
            continue
        with open(p) as f:
            items = json.load(f)
        for item in items:
            img = item.get("img_path", item.get("image", None))
            records.append({
                "question": item["question"],
                "answer": str(item["answer"]),
                "steps": _split_solution(item.get("solution", "")),
                "image_path": f"images/{Path(img).name}" if img else None,
                "_raw_image_src": str(source_dir / img) if img else None,
            })
    return records


def load_math_vision(source_dir: Path):
    """
    MATH-Vision (https://mathvision-cuhk.github.io/).
    Expects data.json and an images/ folder.
    """
    data_file = source_dir / "data.json"
    with open(data_file) as f:
        items = json.load(f)
    records = []
    for item in items:
        img = item.get("image", None)
        records.append({
            "question": item["problem"],
            "answer": str(item["answer"]),
            "steps": _split_solution(item.get("solution", "")),
            "image_path": f"images/{img}" if img else None,
            "_raw_image_src": str(source_dir / "images" / img) if img else None,
        })
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _split_solution(solution: str):
    """Split a free-text solution into a list of steps."""
    if not solution:
        return [""]
    # Split on newlines; fall back to treating the whole string as one step.
    steps = [s.strip() for s in solution.split("\n") if s.strip()]
    return steps if steps else [solution]


def _copy_images(records, output_dir: Path):
    """Copy referenced images into output_dir/images/."""
    img_dir = output_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    for rec in records:
        src = rec.pop("_raw_image_src", None)
        if src and Path(src).exists():
            dst = output_dir / rec["image_path"]
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                shutil.copy2(src, dst)


def _split_records(records, val_ratio=0.05, test_ratio=0.1, seed=42):
    """Random split into train / val / test."""
    rng = random.Random(seed)
    rng.shuffle(records)
    n = len(records)
    n_test = max(1, int(n * test_ratio))
    n_val = max(1, int(n * val_ratio))
    test = records[:n_test]
    val = records[n_test: n_test + n_val]
    train = records[n_test + n_val:]
    return train, val, test


ADAPTERS = {
    "mathvista": load_mathvista,
    "geoqa": load_geoqa,
    "math_vision": load_math_vision,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(ADAPTERS.keys()), required=True,
                        help="Which dataset adapter to use.")
    parser.add_argument("--source_dir", type=Path, required=True,
                        help="Path to the raw downloaded dataset.")
    parser.add_argument("--output_dir", type=Path, required=True,
                        help="Where to write train/val/test.json and images/.")
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--test_ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.dataset} from {args.source_dir} …")
    records = ADAPTERS[args.dataset](args.source_dir)
    print(f"  Loaded {len(records)} records.")

    # Copy images before removing the internal _raw_image_src field
    print("Copying images …")
    _copy_images(records, args.output_dir)

    # Split
    train, val, test = _split_records(records, args.val_ratio, args.test_ratio, args.seed)
    print(f"  train={len(train)}  val={len(val)}  test={len(test)}")

    for split, data in [("train", train), ("val", val), ("test", test)]:
        out_path = args.output_dir / f"{split}.json"
        with open(out_path, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"  Wrote {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()