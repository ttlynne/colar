"""
加载所有 .arrow 文件，筛选 source 为数学类的数据，
并保存为一个合并的 Arrow IPC Stream 文件。
"""

import os
import glob
from collections import Counter

import pyarrow as pa
import pyarrow.ipc as ipc
from torch.utils.data import Dataset, DataLoader


# ── 配置 ──────────────────────────────────────────────────────────────────────
DATA_DIR    = "/mnt/a100_2_data3/tangyueling/DATA/MMFineReason-SFT-123K/train"
OUTPUT_DIR  = "/mnt/a100_2_data3/tangyueling/DATA/MMFineReason-SFT-123K-Math/train"
OUTPUT_FILE = "math_merged.arrow"
BATCH_SIZE  = 512
NUM_WORKERS = 4

# 保留的数学类 source（按需增删）
MATH_SOURCES = {
    "MMR1",
    "WaltonColdStart",
    "ViRL39K",
    "Euclid30K",
    "MMK12",
    "FineVision-geo170k(qa)",
    "FineVision-geometry3k(mathv360k)",
    "mmopenr1-8k",
    "WeMath2-Pro",
    "WeMath2-Standard",
    "WeMath2-SFT",
    "BMMR",
}


# ── 工具：自动识别格式 ────────────────────────────────────────────────────────
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
    try:
        import pyarrow.parquet as pq
        return pq.read_table(fp)
    except Exception:
        pass
    raise ValueError(f"无法识别文件格式: {fp}")


# ── Dataset ───────────────────────────────────────────────────────────────────
class ArrowDataset(Dataset):
    def __init__(self, arrow_files: list):
        self.rows = []
        for fp in arrow_files:
            try:
                table = read_arrow_file(fp)
            except ValueError as e:
                print(f"[WARN] {e}，已跳过。")
                continue
            if "source" not in table.schema.names:
                print(f"[WARN] 'source' 字段不存在: {os.path.basename(fp)}")
                continue
            batch = table.to_pydict()
            keys = list(batch.keys())
            n = len(table)
            for i in range(n):
                self.rows.append({k: batch[k][i] for k in keys})
            print(f"  已加载: {os.path.basename(fp)}  ({n:,} 行)")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


def collate_fn(batch):
    return batch


# ── 主逻辑 ────────────────────────────────────────────────────────────────────
def main():
    # 1. 找文件
    arrow_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.arrow")))
    if not arrow_files:
        raise FileNotFoundError(f"在 {DATA_DIR} 下未找到 .arrow 文件")
    print(f"共找到 {len(arrow_files)} 个 .arrow 文件\n")

    # 2. 加载
    dataset = ArrowDataset(arrow_files)
    dataloader = DataLoader(
        dataset,
        batch_size  = BATCH_SIZE,
        shuffle     = False,
        num_workers = NUM_WORKERS,
        collate_fn  = collate_fn,
    )
    print(f"\n总行数: {len(dataset):,}，开始筛选数学数据...\n")

    # 3. 筛选
    all_rows = []
    source_counter = Counter()

    for batch in dataloader:
        for row in batch:
            src = row.get("source", "")
            if src in MATH_SOURCES:
                all_rows.append(row)
                source_counter[src] += 1

    total_kept = len(all_rows)
    print(f"筛选完毕，保留 {total_kept:,} 条数学数据，共 {len(source_counter)} 种 source\n")

    # 4. 保存为单个合并文件
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    keys = list(all_rows[0].keys())
    col_data = {k: [r[k] for r in all_rows] for k in keys}
    merged_table = pa.table(col_data)
    out_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    with pa.OSFile(out_path, "wb") as sink:
        writer = ipc.new_stream(sink, merged_table.schema)
        writer.write_table(merged_table)
        writer.close()
    print(f"已保存: {out_path}  ({total_kept:,} 行)")

    # 5. 统计输出
    print(f"\n{'source 类型':<50}  {'数量':>8}")
    print("-" * 62)
    for src, cnt in source_counter.most_common():
        print(f"{src:<50}  {cnt:>8,}")
    print("-" * 62)
    print(f"{'合计':<50}  {total_kept:>8,}")


if __name__ == "__main__":
    main()