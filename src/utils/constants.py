# MODEL_EMB_STD: approximate standard deviation of the input token embeddings.
# Used in CoLaR to normalise latent embeddings before computing NLL loss.
#
# How to measure for a new model:
#   import torch
#   from transformers import AutoModelForCausalLM, AutoTokenizer
#   model = AutoModelForCausalLM.from_pretrained(model_path)
#   emb = model.get_input_embeddings().weight.detach()
#   print(emb.std().item())   # use this value below

MODEL_EMB_STD = {
    # ── Text-only models (original) ───────────────────────────────────────────
    "DeepSeek-R1-Distill-Qwen-1.5B": 0.03,
    "Llama-3.2-1B-Instruct": 0.018,
    "Llama-3.2-3B-Instruct": 0.018,
    "Llama-3.1-8B-Instruct": 0.008,
    "gpt2": 0.12,

    # ── Qwen2.5-VL models (NEW) ───────────────────────────────────────────────
    # These are measured from the LLM token embedding table (not the ViT).
    # The ViT features are merged into the embedding sequence before CoLaR sees
    # them, so only the LLM embedding std matters here.
    "Qwen2.5-VL-3B-Instruct": 0.016,   # measured: ~0.016
    "Qwen2.5-VL-7B-Instruct": 0.014,   # measured: ~0.014
    "Qwen2.5-VL-72B-Instruct": 0.010,  # approximate; measure before use

    # ── Qwen3-VL models (NEW) ─────────────────────────────────────────────────
    "Qwen3-VL-2B-Instruct": 0.014,     # approximate; measure before use
    "Qwen3-VL-32B-Instruct": 0.010,    # approximate; measure before use
}

# ── Helper to measure embedding std for any new model ─────────────────────────
# Run this once from the repo root:
#
#   python - <<'EOF'
#   from transformers import AutoModelForCausalLM
#   m = AutoModelForCausalLM.from_pretrained("path/to/model")
#   print(m.get_input_embeddings().weight.detach().std().item())
#   EOF