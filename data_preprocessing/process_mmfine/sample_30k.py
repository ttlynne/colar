"""
从 MMFineReason-SFT-123K-Math/train 下所有 .arrow 文件中
随机抽取 30k 条数据，保存为单个 Arrow 文件。
"""

import os
import glob
import random
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as ipc
from torch.utils.data import Dataset, DataLoader


# ══════════════════════════════════════════════════════════════════════════════
# 配置
# ══════════════════════════════════════════════════════════════════════════════
DATA_DIR    = "/mnt/a100_2_data3/tangyueling/DATA/MMFineReason-SFT-123K-Math/train"
OUTPUT_DIR  = "/mnt/a100_2_data3/tangyueling/DATA/MMFineReason-SFT-123K-Math-30K"
OUTPUT_FILE = "math_30k.arrow"
SAMPLE_SIZE = 30_000
RANDOM_SEED = 63

LOADER_BATCH = 1024
NUM_WORKERS  = 4


# ══════════════════════════════════════════════════════════════════════════════
# 读取 Arrow（自动识别格式）
# ══════════════════════════════════════════════════════════════════════════════
def read_arrow_file(fp: str) -> pa.Table:
    try:
        with pa.memory_map(fp, "r") as src:
            return ipc.open_file(src).read_all()
    except pa.lib.ArrowInvalid:
        pass
    try:
        with pa.memory_map(fp, "r") as src:
            return ipc.open_stream(src).read_all()
    except pa.lib.ArrowInvalid:
        pass
    raise ValueError(f"无法识别文件格式: {fp}")


# ══════════════════════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════════════════════
class ArrowDataset(Dataset):
    def __init__(self, arrow_files: list[str]):
        self.rows = []
        for fp in arrow_files:
            table = read_arrow_file(fp)
            d     = table.to_pydict()
            keys  = list(d.keys())
            n     = len(table)
            for i in range(n):
                self.rows.append({k: d[k][i] for k in keys})
            print(f"  已加载: {Path(fp).name}  ({n:,} 行)")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


# ══════════════════════════════════════════════════════════════════════════════
# 主逻辑
# ══════════════════════════════════════════════════════════════════════════════
def main():
    # 1. 找文件
    arrow_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.arrow")))
    if not arrow_files:
        raise FileNotFoundError(f"在 {DATA_DIR} 下未找到 .arrow 文件")
    print(f"共找到 {len(arrow_files)} 个 .arrow 文件\n")

    # 2. 用 DataLoader 加载所有数据
    dataset    = ArrowDataset(arrow_files)
    dataloader = DataLoader(
        dataset,
        batch_size  = LOADER_BATCH,
        shuffle     = False,
        num_workers = NUM_WORKERS,
        collate_fn  = list,
    )
    total = len(dataset)
    print(f"\n总样本数: {total:,}")

    if SAMPLE_SIZE > total:
        raise ValueError(f"抽样数量 {SAMPLE_SIZE:,} 超过总数据量 {total:,}")

    # 3. 通过 DataLoader 收集所有行
    all_rows = []
    for batch in dataloader:
        all_rows.extend(batch)
    print(f"DataLoader 加载完成，共 {len(all_rows):,} 条\n")

    # 4. 随机抽样
    random.seed(RANDOM_SEED)
    sampled_rows = random.sample(all_rows, SAMPLE_SIZE)
    print(f"随机抽取: {SAMPLE_SIZE:,} 条（seed={RANDOM_SEED}）")

    # 5. 统计抽样后各 source 分布
    from collections import Counter
    source_counter = Counter(r.get("source", "N/A") for r in sampled_rows)
    print(f"\n{'source 类型':<50}  {'数量':>8}  {'占比':>7}")
    print("-" * 70)
    for src, cnt in source_counter.most_common():
        print(f"{src:<50}  {cnt:>8,}  {cnt/SAMPLE_SIZE*100:>6.2f}%")
    print("-" * 70)
    print(f"{'合计':<50}  {SAMPLE_SIZE:>8,}  {'100.00%':>7}\n")

    # 6. 保存
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    keys     = list(sampled_rows[0].keys())
    col_data = {k: [r[k] for r in sampled_rows] for k in keys}
    table    = pa.table(col_data)
    out_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)

    with pa.OSFile(out_path, "wb") as sink:
        writer = ipc.new_stream(sink, table.schema)
        writer.write_table(table)
        writer.close()

    print(f"✅ 已保存: {out_path}  ({SAMPLE_SIZE:,} 条)")


if __name__ == "__main__":
    main()