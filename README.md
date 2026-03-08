# CoLaR-VL: Adding Multimodal (Image + Text) Support to CoLaR

This folder contains the **minimal set of files** you need to modify in the
[CoLaR repository](https://github.com/xiaomi-research/colar) to support
image+text inputs using **Qwen2.5-VL** (or Qwen3-VL) as the backbone.

---

## File map

| This file | Replaces in repo | What changed |
|---|---|---|
| `model_base.py` | `src/models/model_base.py` | VL model loading, `prepare_inputs_vl()`, VL-aware `latent_generate()` |
| `colar.py` | `src/models/colar.py` | Passes `images` through SFT forward + RL rollout; VL position_ids handling |
| `qsa.py` | `src/datasets/qsa.py` | Loads `image_path` from JSON, lazy PIL load, custom `collate_fn` |
| `colar_vl.yaml` | `src/configs/models/colar_vl.yaml` (new) | Qwen2.5-VL-3B backbone, VL LoRA target_modules, smaller batch size |
| `constants.py` | `src/utils/constants.py` | Added `MODEL_EMB_STD` entries for Qwen2.5-VL / Qwen3-VL |
| `mathvision.py` | `data_preprocessing/mathvision.py` (new) | Preprocessing adapters for MathVista / GeoQA / MATH-Vision |

---

## Installation

```bash
# In addition to the original requirements.txt:
pip install qwen-vl-utils          # Qwen VL image processing utilities
pip install Pillow                 # PIL image loading (usually already present)
# transformers >= 4.49 is required for Qwen2.5-VL
pip install "transformers>=4.49"
```

---

## Step-by-step integration

### 1. Copy modified files into the repo

```bash
cp model_base.py   path/to/colar/src/models/model_base.py
cp colar.py        path/to/colar/src/models/colar.py
cp qsa.py          path/to/colar/src/datasets/qsa.py
cp constants.py    path/to/colar/src/utils/constants.py
cp colar_vl.yaml   path/to/colar/src/configs/models/colar_vl.yaml
cp mathvision.py   path/to/colar/data_preprocessing/mathvision.py
```

### 2. Download Qwen2.5-VL

```bash
# Place the model at  <workspace>/models/llms/Qwen2.5-VL-3B-Instruct/
huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct \
    --local-dir <workspace>/models/llms/Qwen2.5-VL-3B-Instruct
```

### 3. Measure embedding std (important!)

```bash
python - <<'EOF'
from transformers import Qwen2_5_VLForConditionalGeneration
m = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "<workspace>/models/llms/Qwen2.5-VL-3B-Instruct"
)
print(m.get_input_embeddings().weight.detach().std().item())
EOF
# Update the value for "Qwen2.5-VL-3B-Instruct" in src/utils/constants.py
```

### 4. Prepare a multimodal dataset

```bash
python data_preprocessing/mathvision.py \
    --dataset mathvista \
    --source_dir /path/to/raw/mathvista \
    --output_dir <workspace>/datasets/text_reasoning/mathvista
```

JSON format for manual datasets:
```json
[
  {
    "question": "In the figure, triangle ABC has AB=5, BC=3. Find AC.",
    "answer": "4",
    "steps": ["By the Pythagorean theorem ...", "Therefore AC = 4"],
    "image_path": "images/geom_001.png"
  }
]
```
`image_path` is **relative to the dataset directory**.  
Records without `image_path` (or with `null`) are treated as text-only.

### 5. Train

```bash
python run.py \
    --devices=all \
    --model=colar_vl \
    --dataset=qsa \
    --do_test \
    dataset_name=mathvista \
    model_id=Qwen2.5-VL-3B-Instruct \
    batch_size=32 \
    max_compression_factor=5 \
    compression_factor=5 \
    max_new_tokens=16 \
    max_epochs=50
```

---

## Architecture overview

```
Image ──► ViT (frozen) ──► visual tokens ─┐
                                           ├──► [question_embeds | steps_ec | answer] ──► LLM + LatentHead
Text  ──► LLM Token Embed ────────────────┘
```

The key change vs. the original CoLaR is inside `prepare_inputs_vl()` in
`model_base.py`.  It calls `processor.apply_chat_template()` and then
`model.model.prepare_inputs_embeds()` (Qwen2.5-VL's built-in method) which:
1. Runs the ViT on the image pixels.
2. Merges the visual token features into the text embedding sequence at the
   positions marked by `<image_pad>` special tokens.

The returned `inputs_embeds` tensor is then used exactly like the original
`question_inputs_embeds` in CoLaR's SFT and RL code paths.

---

## Notes on position_ids (M-RoPE)

Qwen2.5-VL uses **Multimodal RoPE (M-RoPE)** for position encoding.
`prepare_inputs_vl()` returns the `position_ids` computed by the VL model
for the question+image prefix.  For the reasoning (`steps`) and `answer`
segments we extend them linearly:

```python
extra_pos = q_last_pos + torch.arange(1, n_extra + 1)
position_ids = torch.cat([question_position_ids, extra_pos], dim=1)
```

This is correct because the reasoning and answer tokens are plain text — they
do not require 2-D spatial position encoding.

---

## Reducing GPU memory

- Use `batch_size=32` (or lower) — images consume much more memory than text.
- Add `torch_dtype=torch.bfloat16` to the `from_pretrained` call in `model_base.py`.
- Enable `attn_implementation="flash_attention_2"` if flash-attn is installed.
- Freeze the ViT by adding after model load:
  ```python
  for n, p in self.llm.named_parameters():
      if "visual" in n:
          p.requires_grad_(False)
  ```
  (The LoRA `target_modules` already excludes ViT layers, so this is only
  needed if you want to make sure no accidental gradients flow through it.)