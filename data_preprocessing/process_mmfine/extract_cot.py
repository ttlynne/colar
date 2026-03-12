"""
用 vLLM 多卡并行从 qwen3vl_235b_thinking_response 字段提取计算步骤，
写入新的 cot 字段，保存为单个 Arrow 文件。

改动要点（对比 transformers 版）：
  - 用 vLLM LLM + tensor_parallel_size 替换 transformers，自动使用全部 GPU
  - vLLM 内置 continuous batching，一次性提交所有待推理 prompt，吞吐最大化
  - 去掉 INFER_BATCH_SIZE 手动分批，改为 VLLM_BATCH_SIZE（提交给 vLLM 的块大小）
  - 断点续传、DataLoader、Arrow 读写逻辑保持不变
"""

import os
import glob
import json
import time
import traceback
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as ipc
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


# ══════════════════════════════════════════════════════════════════════════════
# 配置
# ══════════════════════════════════════════════════════════════════════════════
DATA_DIR    = "/mnt/a100_2_data3/tangyueling/DATA/MMFineReason-SFT-123K-Math-30K"
OUTPUT_PATH = "/mnt/a100_2_data3/tangyueling/DATA/MMFineReason-SFT-123K-Math-CoT/math_with_cot.arrow"
CKPT_PATH   = OUTPUT_PATH + ".ckpt.jsonl"

MODEL_PATH  = "/mnt/a100_2_data3/tangyueling/MODELS/Qwen/Qwen3-32B"

# ── vLLM 关键参数 ──────────────────────────────────────────────────────────
TENSOR_PARALLEL_SIZE = 4      # ← 你的 GPU 数量（2 或 4），自动跨卡张量并行
VLLM_BATCH_SIZE      = 512    # 每次提交给 vLLM 的 prompt 数量（越大吞吐越高）
MAX_NEW_TOKENS       = 512
GPU_MEMORY_UTILIZATION = 0.90 # vLLM 显存占用比例，可调低至 0.85 避免 OOM

# ── DataLoader 参数 ────────────────────────────────────────────────────────
LOADER_BATCH = 1024
NUM_WORKERS  = 4


# ══════════════════════════════════════════════════════════════════════════════
# Prompt 模板
# ══════════════════════════════════════════════════════════════════════════════
SYSTEM_PROMPT = (
    "You are a math reasoning extractor. "
    "Given a solution text, extract ONLY the arithmetic/algebraic calculation steps. "
    "Output them in the format: <<expr=result>> separated by spaces. "
    "For example: <<600*30/100=180>> <<600*10/100=60>> <<180+60=240>> <<600-240=360>> "
    "Output NOTHING else — no explanation, no punctuation outside the brackets."
)

def make_messages(response_text: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": response_text},
    ]


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
class MathDataset(Dataset):
    def __init__(self, arrow_files: list[str]):
        self.rows = []
        for fp in arrow_files:
            table = read_arrow_file(fp)
            d = table.to_pydict()
            keys = list(d.keys())
            n = len(table)
            for i in range(n):
                self.rows.append({k: d[k][i] for k in keys})
            print(f"  已加载: {Path(fp).name}  ({n:,} 行)")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


# ══════════════════════════════════════════════════════════════════════════════
# 断点续传
# ══════════════════════════════════════════════════════════════════════════════
def load_checkpoint(ckpt_path: str) -> dict[int, str]:
    done = {}
    if not os.path.exists(ckpt_path):
        return done
    with open(ckpt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            done[obj["idx"]] = obj["cot"]
    return done


def save_checkpoint_batch(ckpt_path: str, items: list[tuple[int, str]]):
    """批量追加写入，减少 IO 次数"""
    os.makedirs(os.path.dirname(ckpt_path) or ".", exist_ok=True)
    with open(ckpt_path, "a", encoding="utf-8") as f:
        for idx, cot in items:
            f.write(json.dumps({"idx": idx, "cot": cot}, ensure_ascii=False) + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# 主逻辑
# ══════════════════════════════════════════════════════════════════════════════
def main():
    # ── 1. 加载数据 ──────────────────────────────────────────────────────────
    arrow_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.arrow")))
    if not arrow_files:
        raise FileNotFoundError(f"在 {DATA_DIR} 下未找到 .arrow 文件")
    print(f"\n共找到 {len(arrow_files)} 个 .arrow 文件")

    dataset    = MathDataset(arrow_files)
    dataloader = DataLoader(
        dataset,
        batch_size  = LOADER_BATCH,
        shuffle     = False,
        num_workers = NUM_WORKERS,
        collate_fn  = list,
    )
    total = len(dataset)
    print(f"总样本数: {total:,}\n")

    # ── 2. 断点续传 ───────────────────────────────────────────────────────────
    done_map = load_checkpoint(CKPT_PATH)
    print(f"断点续传: 已完成 {len(done_map):,} 条，剩余 {total - len(done_map):,} 条\n")

    # ── 3. 初始化 tokenizer（仅用于 apply_chat_template）────────────────────
    print("加载 tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # ── 4. 初始化 vLLM（多卡张量并行）───────────────────────────────────────
    print(f"初始化 vLLM (tensor_parallel_size={TENSOR_PARALLEL_SIZE}) ...")
    llm = LLM(
        model                  = MODEL_PATH,
        tensor_parallel_size   = TENSOR_PARALLEL_SIZE,
        gpu_memory_utilization = GPU_MEMORY_UTILIZATION,
        dtype                  = "bfloat16",
        trust_remote_code      = True,
        max_model_len          = 4096,   # 输入最长 token 数
        enforce_eager          = False,  # 开启 CUDA graph，加速推理
    )
    sampling_params = SamplingParams(
        temperature     = 0.0,           # greedy，输出稳定
        max_tokens      = MAX_NEW_TOKENS,
        skip_special_tokens = True,
    )
    print("vLLM 初始化完成\n")

    # ── 5. 收集所有待推理样本 ────────────────────────────────────────────────
    # 先遍历一遍，把所有未完成的 (idx, prompt_str) 收集好，再批量提交给 vLLM
    print("收集待推理样本，生成 prompt ...")
    pending_indices: list[int] = []
    pending_prompts: list[str] = []

    for row in dataset.rows:
        idx = len(pending_indices) + sum(1 for i in range(total) if i < len(pending_indices))

    # 重写：直接遍历索引
    pending_indices = []
    pending_prompts = []
    for i, row in enumerate(dataset.rows):
        if i in done_map:
            continue
        response_text = row.get("qwen3vl_235b_thinking_response", "") or ""
        messages      = make_messages(response_text)
        prompt_str    = tokenizer.apply_chat_template(
            messages,
            tokenize            = False,
            add_generation_prompt = True,
            enable_thinking     = False,  # 关闭思考模式，直接输出结果
        )
        pending_indices.append(i)
        pending_prompts.append(prompt_str)

    print(f"共 {len(pending_prompts):,} 条需要推理\n")

    # ── 6. 分批提交 vLLM 推理（continuous batching 自动并行）────────────────
    os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
    t0 = time.time()

    for batch_start in range(0, len(pending_prompts), VLLM_BATCH_SIZE):
        batch_end     = min(batch_start + VLLM_BATCH_SIZE, len(pending_prompts))
        batch_indices = pending_indices[batch_start:batch_end]
        batch_prompts = pending_prompts[batch_start:batch_end]

        try:
            outputs = llm.generate(batch_prompts, sampling_params)
            cots    = [out.outputs[0].text.strip() for out in outputs]
        except Exception:
            print(f"[ERROR] vLLM 推理失败 batch {batch_start}-{batch_end}:\n{traceback.format_exc()}")
            cots = [""] * len(batch_prompts)

        # 批量写断点
        batch_ckpt = list(zip(batch_indices, cots))
        save_checkpoint_batch(CKPT_PATH, batch_ckpt)
        for idx, cot in batch_ckpt:
            done_map[idx] = cot

        elapsed   = time.time() - t0
        finished  = len(done_map)
        speed     = finished / elapsed if elapsed > 0 else 0
        remaining = total - finished
        eta       = remaining / speed if speed > 0 else float("inf")
        print(
            f"  进度: {finished:>6,}/{total:,} "
            f"({100 * finished / total:.1f}%)  "
            f"速度: {speed:.1f} 条/s  ETA: {eta / 60:.1f} min"
        )

    # ── 7. 打印前 10 条处理前后对比 ──────────────────────────────────────────
    print("\n" + "═" * 80)
    print("前 10 条处理结果预览")
    print("═" * 80)
    for i, row in enumerate(dataset.rows[:10]):
        original         = (row.get("qwen3vl_235b_thinking_response", "") or "").strip()
        cot              = done_map.get(i, "")
        original_preview = original[:300] + ("..." if len(original) > 300 else "")
        print(f"\n┌─ 样本 #{i+1} (idx={i}) " + "─" * 55)
        print(f"│ source : {row.get('source', 'N/A')}")
        print(f"│")
        print(f"│ [处理前] qwen3vl_235b_thinking_response (前300字):")
        for line in original_preview.splitlines():
            print(f"│   {line}")
        print(f"│")
        print(f"│ [处理后] cot:")
        print(f"│   {cot}")
        print("└" + "─" * 70)
    print()

    # ── 8. 组装并写入最终 Arrow 文件 ─────────────────────────────────────────
    print("所有样本推理完成，写入输出文件...")

    all_rows = dataset.rows
    for i, row in enumerate(all_rows):
        row["cot"] = done_map.get(i, "")

    keys     = list(all_rows[0].keys())
    col_data = {k: [r[k] for r in all_rows] for k in keys}
    table    = pa.table(col_data)

    with pa.OSFile(OUTPUT_PATH, "wb") as sink:
        writer = ipc.new_stream(sink, table.schema)
        writer.write_table(table)
        writer.close()

    print(f"\n✅ 完成！已保存到: {OUTPUT_PATH}")
    print(f"   总样本: {len(all_rows):,}  字段: {keys}")


if __name__ == "__main__":
    main()