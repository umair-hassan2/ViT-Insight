# ViT-Insight

Interpretability toolkit and benchmark for **attention-based explanations in Vision Transformers**.
Visualise attention rollout, per-layer and per-head attention for supervised, contrastive (CLIP) and
self-supervised (DINO, DINOv2 ± registers) ViTs through a Python API, a CLI and a Gradio app.

Research plan and roadmap: [`docs/RESEARCH-PLAN.md`](docs/RESEARCH-PLAN.md).

## Getting started (run the app locally)

Requires Python ≥3.10 and either [`uv`](https://docs.astral.sh/uv/) (recommended) or `pip`.

1. **Clone the repository**
   ```bash
   git clone https://github.com/umair-hassan2/ViT-Insight.git
   cd ViT-Insight
   ```

2. **Install dependencies**
   ```bash
   uv sync --extra app
   # or, without uv:
   pip install -e ".[app]"
   ```

3. **Run the Gradio app**
   ```bash
   uv run python app/gradio_app.py
   # or, without uv (inside the environment pip installed into):
   python app/gradio_app.py
   ```

4. **Open the app** at http://127.0.0.1:7860
   - Pick a model from the dropdown (e.g. `google/vit-base-patch16-224`).
   - Upload an image.
   - For CLIP/SigLIP models, enter 1–5 comma-separated text labels.
   - Choose a view (**Attention Rollout**, **Per-layer GIF**, or **Head grid**) and click **Explain**.

The first run for a given model downloads its weights from Hugging Face and caches them under
`~/.cache/huggingface`, so it is slower than later runs.

## Use from the command line

```bash
uv run vit-insight models                                   # list supported checkpoints
uv run vit-insight explain --model google/vit-base-patch16-224 --image cat.jpg --gif layers.gif
uv run vit-insight explain --model openai/clip-vit-base-patch16 --image cat.jpg \
    --labels "a photo of a cat,a photo of a dog" --head-fusion max --discard-ratio 0.9
```

## Use as a library

```python
from PIL import Image
from vit_insight import load_model, run_forward
from vit_insight.methods import attention_rollout
from vit_insight.viz import rollout_overlay, head_grid

lm = load_model("facebook/dinov2-with-registers-base")      # cached, eager attention, cuda/mps/cpu
res = run_forward(lm, Image.open("cat.jpg"))                # res.attentions: 12 × (1, 12, S, S)
rollout = attention_rollout(res.attentions, head_fusion="mean", discard_ratio=0.0)  # (1, S, S)
rollout_overlay(res, lm.spec).save("rollout.png")
head_grid(res, lm.spec, layer=11).save("heads.png")
```

## Supported models

| Checkpoint | Objective | Prefix tokens |
|---|---|---|
| `google/vit-base-patch16-224`, `timm/vit_base_patch16_224.augreg2_in21k_ft_in1k` | supervised | 1 (CLS) |
| `facebook/deit-base-distilled-patch16-224` | supervised + distillation | 2 (CLS, dist) |
| `openai/clip-vit-base-patch16`, `openai/clip-vit-base-patch32` | contrastive, text-conditioned | 1 |
| `facebook/dino-vitb16`, `facebook/dinov2-base` | self-supervised | 1 |
| `facebook/dinov2-with-registers-base` | self-supervised + 4 registers | 5 |
| `google/siglip-base-patch16-224` | sigmoid contrastive | 0 (no CLS; CLS-row methods disabled) |

Add a checkpoint by appending to `vit_insight/configs/models.yaml` (`prefix_tokens` and `patch_size`
are what the token layout needs; the patch grid is derived from the processed image size).

## Method

Attention rollout follows Abnar & Zuidema (ACL 2020): per layer, heads are fused
(`mean`/`max`/`min`), optionally the lowest `discard_ratio` fraction of entries per row is zeroed,
the residual connection is modelled as Ã_l = rownorm(½A_l + ½I), and layers compose as
Ã_L · … · Ã_1. The CLS row of the result over patch tokens is the explanation.
`legacy_rollout` keeps the pre-0.2 implementation (no residual term, reversed product) for ablation;
on ViT-B/16 the two correlate at only ≈0.28.

## Development

```bash
uv sync --extra dev --extra app
uv run ruff check . && uv run pytest -q    # tests run on a tiny random ViT, no downloads
```

## Reference

Abnar, S., & Zuidema, W. (2020). *Quantifying Attention Flow in Transformers.* ACL. https://arxiv.org/abs/2005.00928

MIT License.
