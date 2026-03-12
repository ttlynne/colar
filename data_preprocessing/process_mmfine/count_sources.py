"""
加载指定目录下所有 .arrow 文件，统计 "source" 字段的各类型数量。
自动识别 Arrow IPC File / IPC Stream / Parquet 格式。
"""

import os
import glob
from collections import Counter

import pyarrow as pa
import pyarrow.ipc as ipc
from torch.utils.data import Dataset, DataLoader


# ── 配置 ──────────────────────────────────────────────────────────────────────
DATA_DIR    = "/mnt/a100_2_data3/tangyueling/DATA/MMFineReason-SFT-123K/train"
BATCH_SIZE  = 512
NUM_WORKERS = 4


# ── 工具：自动识别格式并读取 Arrow 文件 ──────────────────────────────────────
def read_arrow_file(fp: str) -> pa.Table:
    """
    依次尝试：
      1. IPC File   (ipc.open_file)   —— 有 magic footer
      2. IPC Stream (ipc.open_stream) —— 常见于 HuggingFace datasets
      3. Parquet    (parquet.read_table)
    """
    # 方式 1: IPC File
    try:
        with pa.memory_map(fp, "r") as src:
            return ipc.open_file(src).read_all()
    except pa.lib.ArrowInvalid:
        pass

    # 方式 2: IPC Stream
    try:
        with pa.memory_map(fp, "r") as src:
            return ipc.open_stream(src).read_all()
    except pa.lib.ArrowInvalid:
        pass

    # 方式 3: Parquet
    try:
        import pyarrow.parquet as pq
        return pq.read_table(fp)
    except Exception:
        pass

    raise ValueError(f"无法识别文件格式: {fp}")


# ── Dataset ───────────────────────────────────────────────────────────────────
class ArrowDataset(Dataset):
    def __init__(self, arrow_files: list):
        self.records = []
        for fp in arrow_files:
            try:
                table = read_arrow_file(fp)
            except ValueError as e:
                print(f"[WARN] {e}，已跳过。")
                continue

            if "source" not in table.schema.names:
                print(f"[WARN] 'source' 字段不存在于 {os.path.basename(fp)}，"
                      f"可用字段: {table.schema.names}")
                continue

            self.records.extend(table.column("source").to_pylist())
            print(f"  已加载: {os.path.basename(fp)}  ({len(table):,} 行)")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]


# ── 主逻辑 ────────────────────────────────────────────────────────────────────
def main():
    pattern     = os.path.join(DATA_DIR, "*.arrow")
    arrow_files = sorted(glob.glob(pattern))
    if not arrow_files:
        raise FileNotFoundError(f"在 {DATA_DIR} 下未找到任何 .arrow 文件")
    print(f"共找到 {len(arrow_files)} 个 .arrow 文件\n")

    dataset = ArrowDataset(arrow_files)
    print(f"\n数据集总行数: {len(dataset):,}\n")

    dataloader = DataLoader(
        dataset,
        batch_size  = BATCH_SIZE,
        shuffle     = False,
        num_workers = NUM_WORKERS,
        collate_fn  = list,
    )

    counter = Counter()
    for batch in dataloader:
        counter.update(batch)

    total = sum(counter.values())
    print(f"\n{'source 类型':<50}  {'数量':>10}  {'占比':>8}")
    print("-" * 74)
    for src, cnt in counter.most_common():
        pct = cnt / total * 100
        print(f"{str(src):<50}  {cnt:>10,}  {pct:>7.2f}%")
    print("-" * 74)
    print(f"{'合计':<50}  {total:>10,}  {'100.00%':>8}")
    print(f"\n共 {len(counter)} 种 source 类型")


if __name__ == "__main__":
    main()