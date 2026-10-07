# ViT-Insight → Research Project Plan

**Prepared:** 2026-10-07, decisions 2026-10-08 · **Repo:** https://github.com/umair-hassan2/ViT-Insight · **Target:** grad-application-ready research project by ~Dec 1, 2026 (8 weeks), stretch to a workshop submission in Feb–Mar 2027.

---

## 0. Bottom line

ViT-Insight today is a 454-line Gradio demo (2 models, 2 views). It is a **tool**, not yet a **research project**: no research question, no quantitative evaluation, no write-up, and the one algorithm it implements (attention rollout) deviates from the paper it cites. The fastest route to something a grad committee reads as research is:

1. **Fix and harden the core** (rollout math, token layout, model caching, tests, packaging) — 2 weeks.
2. **Pick one research question and answer it with numbers.** Recommended: *"How do pretraining objective and register tokens affect the faithfulness of attention-based explanations in ViTs?"* — a controlled study across supervised / CLIP / DINOv2 / DINOv2-with-registers / DeiT, 5–6 explanation methods, standard faithfulness + localization metrics. Nobody has published exactly this slice; the Aug-2026 benchmark by Nishankar et al. varies *architecture* but not *pretraining objective* or *registers*.
3. **Turn the explorer into the interactive companion of that study** (label-conditional CLIP explanations, head-level views, model-vs-model comparison, HF Space).
4. **Write a 6–8 page technical report, post to arXiv (cs.CV) by ~Dec 1**, cite it in SoP/CV. Stretch: ICLR 2027 or CVPR 2027 XAI4CV workshop.

Dev and pilots run on the MacBook M5 (MPS); the full benchmark runs on Colab Pro GPUs.

---

## 1. Audit of the current repo

### 1.1 What exists

| File | Lines | Role |
|---|---|---|
| `src/main.py` | 98 | Gradio Blocks UI, wiring |
| `src/model_manager.py` | 95 | model/processor loading, forward pass |
| `src/inference.py` | 87 | rollout + per-layer heatmaps + GIF |
| `src/model_config.py` | 41 | YAML model registry |
| `src/model_loader.py` | 14 | deprecated dead code |
| `src/configs/models.yaml` | 8 | 2 models: CLIP ViT-B/32, google/vit-base-patch16-224 |
| `README.md` | 104 | features, screenshots, Abnar & Zuidema citation |

No tests, no LICENSE, no CI, no `pyproject.toml`, no pinned deps, 0 stars, last commit 2025-10-12.

### 1.2 Correctness issues (fix first — these would be caught by any reviewer)

| # | Issue | Where | Why it matters |
|---|---|---|---|
| C1 | **Rollout omits the residual identity term.** Abnar & Zuidema define Ã_l = normalize(½A_l + ½I). Code uses A_l directly. | `inference.py:attention_rollout` | The README claims to implement the cited paper; it does not. Without the identity term rollout collapses toward uniform/noisy maps. |
| C2 | **Rollout multiplication order is reversed.** Code computes A_1·A_2·…·A_L; the paper (and jacobgil/vit-explain, `result = a @ result`) computes Ã_L·…·Ã_1. Matrix products do not commute. | `inference.py:18` | Taking row 0 of the wrong product gives a different map. |
| C3 | **"Per-label patch alignment" is not implemented.** CLIP's vision tower never sees the text, so `vision_model_output.attentions` is identical for every label. Only the predicted-label string changes. | `model_manager.py:run_inference`, README "Multi-label support" | README over-claims. Label-conditional maps need a gradient-based method (Chefer et al. 2021 generic attention explainability) or patch-token × text-embedding similarity. |
| C4 | **Patch grid inferred via `sqrt(num_tokens-1)`; assumes exactly one non-patch token.** | `inference.py` (both branches) | Breaks on DeiT-distilled (CLS+dist = 2 extra), DINOv2-with-registers (CLS+4 regs = 5 extra), SigLIP (no CLS). Use `config.image_size // config.patch_size` and a per-model `num_prefix_tokens`. |
| C5 | **`output_attentions=True` returns `None` under SDPA attention** in transformers ≥4.36; current PyPI is 5.19.0. Must pass `attn_implementation="eager"`. | `model_manager.py:load` | App will crash with `None` attentions on a fresh install. (HF issues #30978, #41929.) |
| C6 | **Model reloaded from disk/HF on every click.** Comment says "assume internal caching"; there is none. `device` parameter is unused. | `main.py:gradio_interface`, `model_manager.py:load` | 2–10 s latency per run; no GPU/MPS use. |
| C7 | **Image de-normalisation uses global min/max of `pixel_values`.** CLIP/ViT normalise per-channel; min/max over the whole tensor shifts colours. Use `processor.image_mean/std`. | `model_manager.py` | Overlay background looks wrong vs the uploaded image. |
| C8 | README LaTeX for "normalised patch embeddings per layer" describes a feature that does not exist; clone URL points to `your-username/vision-attention-explorer`; app title ≠ repo name. | `README.md` | Credibility. |

### 1.3 Engineering gaps

- No unit tests (rollout against a hand-computed 3-token example; token-layout per model; shape checks).
- No `pyproject.toml` / lockfile; `requirements.txt` has loose lower bounds (`gradio>=3.50` while 6.29.1 is current).
- No `LICENSE` (recommend MIT), no `CITATION.cff`, no CI, no pre-commit/ruff.
- Flat `src/` with `import model_config` style imports — not installable, not importable from a notebook.
- No CLI / Python API; everything is reachable only through the Gradio callback.
- Dead file `model_loader.py`; experimentation notebook was deleted from history (consider restoring as `notebooks/`).

---

## 2. Research framing

### 2.1 Where the field is (Oct 2026)

| Thread | Key works | Relevance to ViT-Insight |
|---|---|---|
| Attention-as-explanation | Abnar & Zuidema 2020 (rollout/flow); Jain & Wallace 2019 vs Wiegreffe & Pinter 2019 | The method the repo implements; known to be class-agnostic and to highlight unrelated tokens. |
| Gradient/relevance methods for transformers | Chefer et al. CVPR 2021 (LRP+grad), Chefer et al. ICCV 2021 (**generic bi-modal → works on CLIP**), AttnLRP (Achtibat et al. ICML 2024, `lxt` 2.1 on PyPI), DAP (Jo et al. Apr 2026, gradient-weighted rollout), HiLRP (Sep 2026) | Baselines the study must include; Chefer-generic is the fix for C3. |
| Artifact / register tokens | Darcet et al. ICLR 2024 "ViTs Need Registers"; Jiang et al. NeurIPS 2025 Spotlight "ViTs Don't Need Trained Registers" (test-time registers; code released) | Directly explains noisy CLIP/DINOv2 attention maps. HF ships `facebook/dinov2-base` **and** `facebook/dinov2-with-registers-base` — a free controlled pair. |
| Benchmarks / evaluation | Nishankar et al. Aug 2026 (13 methods × 8 architectures; rollout "consistent but poorly localizing"); Wu et al. CVPR 2024 "On the Faithfulness of ViT Explanations"; Quantus 0.6.0; insertion/deletion (RISE), pointing game, sanity checks (Adebayo 2018) | Metric stack to reuse. Gap: no benchmark varies **pretraining objective** or **registers** while holding architecture fixed. |
| CLIP-specific interpretability | Gandelsman et al. ICLR 2024 TextSpan (per-head text bases) | Optional "head role" feature for the explorer. |
| Mechanistic tooling | ViT-Prisma (Joseph et al. 2025), SAEs for CLIP/DINO | Out of scope for 8 weeks; mention as future work. |
| Surveys | "Explainability of Vision Transformers: a comprehensive review" (TMLR submission, Feb 2026) | Use for related-work section. |

### 2.2 Research question (recommended)

> **RQ:** Holding architecture fixed (ViT-B/16, 12 layers, 12 heads, 224 px), how do (a) the pretraining objective and (b) the presence of register tokens change the faithfulness and localisation quality of *attention-based* explanations relative to gradient-based ones?

Hypotheses to test:

- **H1 (correctness):** Residual-corrected rollout (Abnar) beats the repo's current uncorrected variant and raw last-layer CLS attention on deletion/insertion AUC. *(Cheap, validates the fix.)*
- **H2 (objective):** Faithfulness of attention-based maps varies more across pretraining objective (supervised vs CLIP vs DINOv2) than across attention-based method choice.
- **H3 (registers):** DINOv2-with-registers yields higher pointing-game accuracy and lower attention entropy than DINOv2 without, for the same explanation method; test-time registers recover most of that gap on CLIP (stretch).
- **H4 (gap):** Gradient-based methods (Chefer-generic, AttnLRP) are less sensitive to objective/registers than attention-based ones — i.e. registers matter mainly if you insist on attention as the explanation.
- **H5 (descriptive):** Per-head mean attention distance profiles (Dosovitskiy 2020 Fig. 11 style) cluster by pretraining objective.

Why this is defensible as "research": it is a controlled, hypothesis-driven, quantitatively evaluated study with a clear gap, reusable code, and an interactive artifact. It does not require training anything.

### 2.3 Alternatives considered

| Option | Pros | Cons | Verdict |
|---|---|---|---|
| **A. Objective × registers faithfulness study (above)** | Clear gap, cheap, no training, strong narrative tied to 2025 NeurIPS spotlight | Needs a careful metric pipeline | **Recommended** |
| B. Label-conditional CLIP explainer (Chefer + TextSpan) as a tool paper | Fixes C3, very demo-able | Mostly integration; weak novelty | Do as a *feature* of A, not the thesis |
| C. Head-role taxonomy across models (attention distance, TextSpan) | Pretty figures, cheap | Descriptive only; Raghu et al. 2021 covered much of it | Fold in as H5 |
| D. Train SAEs on ViT activations | Trendy (mech-interp) | Compute + 8 weeks is too tight; Prisma already ships 80+ SAEs | Future work |

---

## 3. Experimental design

### 3.1 Models (all verified on HF Hub, 2026-10-07)

| Model | HF id | Objective | Prefix tokens | Notes |
|---|---|---|---|---|
| ViT-B/16 supervised | `google/vit-base-patch16-224` | ImageNet-21k→1k supervised | 1 (CLS) | already in repo |
| ViT-B/16 AugReg | `timm/vit_base_patch16_224.augreg2_in21k_ft_in1k` | supervised, stronger aug | 1 | optional 2nd supervised point |
| DeiT-B distilled | `facebook/deit-base-distilled-patch16-224` | supervised + distillation | 2 (CLS, dist) | exercises C4 |
| CLIP ViT-B/16 | `openai/clip-vit-base-patch16` | contrastive image-text | 1 | swap in for B/32 so patch grid = 14×14 like the others |
| CLIP ViT-B/32 | `openai/clip-vit-base-patch32` | contrastive | 1 | keep for continuity |
| DINO ViT-B/16 | `facebook/dino-vitb16` | self-sup (v1) | 1 | famous clean attention maps |
| DINOv2-B/14 | `facebook/dinov2-base` | self-sup (v2) | 1 | known artifact tokens |
| DINOv2-B/14 + registers | `facebook/dinov2-with-registers-base` | self-sup + 4 registers | 5 | **controlled pair** with the row above |
| SigLIP-B/16 | `google/siglip-base-patch16-224` | sigmoid contrastive | 0 (attention pooling) | stretch; needs pooling-head handling |

### 3.2 Explanation methods

| Method | Type | Class-conditional? | Implementation |
|---|---|---|---|
| Raw last-layer CLS attention (mean heads) | attention | no | exists |
| Rollout, uncorrected (current repo) | attention | no | exists — keep as ablation for H1 |
| Rollout, Abnar-corrected (½A+½I, correct order), head fusion mean/max/min, discard ratio | attention | no | fix C1/C2 |
| Attention Flow (max-flow) | attention | no | optional; expensive |
| Grad-rollout / DAP-style gradient weighting | hybrid | yes | jacobgil grad-rollout as reference |
| Chefer generic (ICCV 2021) | relevance+grad | yes, works for CLIP text queries | port from `hila-chefer/Transformer-MM-Explainability` |
| AttnLRP | LRP | yes | `pip install lxt` (2.1) |
| Random / centre-prior baselines | control | — | required by the Aug-2026 benchmark's finding that some metrics can't beat random |

### 3.3 Data

- **ImageNet-1k val subset**: 1,000 images (1 per class) via HF `ILSVRC/imagenet-1k` (gated, free) for faithfulness metrics on the supervised/DeiT models; zero-shot prompts ("a photo of a {class}") for CLIP/SigLIP; k-NN or linear head for DINO/DINOv2 (or evaluate DINO rows on localisation only).
- **ImageNet-S (919-class subset) or PASCAL VOC 2012 val**: segmentation masks for pointing game / energy pointing game / mask IoU. VOC is freely downloadable and works with CLIP zero-shot (20 classes).
- Fixed seed, fixed 224 px resize+centre-crop, all preprocessing via each model's own HF processor.

### 3.4 Metrics (reuse Quantus where possible)

| Dimension | Metric | Notes |
|---|---|---|
| Faithfulness | Deletion AUC ↓, Insertion AUC ↑ (RISE), their difference | 20 perturbation steps, blur or mean baseline; report both baselines |
| Localisation | Pointing game, Energy pointing game, mask IoU@top-k | needs masks (VOC / ImageNet-S) |
| Class sensitivity | Δ map when target class changes (CLIP: swap prompt) | attention methods should score ~0 → quantifies C3 |
| Sanity | Model-parameter randomisation (Adebayo) — Spearman/SSIM vs trained model | catches methods that are just edge detectors |
| Complexity | Map entropy, sparsity (Gini) | ties to register hypothesis H3 |
| Cost | wall-clock per image, #forward/backward passes | one table |

### 3.5 Compute budget

| Item | Count | Est. |
|---|---|---|
| Models | 8 | — |
| Methods | 7 (+2 baselines) | — |
| Images | 5,000 ImageNet-val (faithfulness, 5/class) + ~1,450 VOC val (localisation) | — |
| Forward passes for insertion+deletion | 2 × 20 steps × 5,000 × 10 × 9 ≈ 18 M single images, batched ×20 → ~900 k batches | ViT-B ~0.05 s/batch on an A100 → **~12 h on Colab Pro**; pilot (300 imgs) on the M5 in ~2 h |
| Storage | attribution maps cached as float16 `.npy` | ~8 × 9 × 1,000 × 196 × 2 B ≈ 28 MB |

Run a 300-image pilot on the M5 first to lock the pipeline, then the full grid on Colab Pro (A100/L4), checkpointing per (model, method) so a disconnected runtime resumes.

---

## 4. Tool roadmap (the explorer becomes the paper's companion)

Priority order; P0 = needed for the study, P1 = strengthens the demo, P2 = nice-to-have.

| P | Feature | Detail |
|---|---|---|
| P0 | Correct rollout + head fusion + discard ratio + layer range | fixes C1/C2; UI exposes `head_fusion`, `discard_ratio` |
| P0 | Model registry with `num_prefix_tokens`, `patch_size`, `image_size`, `family` | replaces sqrt hack (C4); models in §3.1 |
| P0 | Eager attention + cached loading + device auto-select (cuda/mps/cpu) | C5, C6 |
| P0 | Python API + CLI (`vit-insight explain --model … --image … --method …`) | so the benchmark script and notebooks reuse the same code as the UI |
| P0 | Metrics module (`vit_insight.eval`) wrapping Quantus + own pointing game | §3.4 |
| P1 | **Label-conditional CLIP maps** via Chefer-generic and patch×text similarity | finally delivers the README's promise (C3) |
| P1 | **Side-by-side model comparison** (same image, N models, same method) | the visual argument for H2/H3 |
| P1 | **Per-head grid** (12×12 thumbnails) + per-head attention-distance plot | H5; "heads" is in the README but not implemented |
| P1 | Artifact-token inspector: token-norm histogram, highlight high-norm patches, toggle "mask artifact tokens" | ties UI to the register story |
| P1 | Deploy to **Hugging Face Spaces** (CPU basic tier is fine with caching) | one-click demo link for the application |
| P1 | Test-time registers toggle for CLIP/DINOv2 (port from `nickjiang2378/test-time-registers`) | H3 second manipulation (promoted 2026-10-08) |
| P2 | TextSpan head labels for CLIP | descriptive |
| P2 | Export: PNG/GIF/NPY + JSON of metrics; shareable permalink | reproducibility |
| P2 | Sanity-check button (randomise weights → compare map) | pedagogical |

---

## 5. Engineering hygiene (do in week 1; cheap, high signal to reviewers)

- Restructure to an installable package:
  ```
  vit_insight/
    __init__.py
    registry.py        # model specs (yaml-backed)
    loading.py         # cached loader, eager attn, device
    tokens.py          # prefix-token / patch-grid handling
    methods/           # rollout.py, raw_attention.py, chefer.py, attnlrp.py, baselines.py
    eval/              # faithfulness.py, localization.py, sanity.py, runner.py
    viz.py             # overlays, GIF, head grid
    cli.py
  app/gradio_app.py
  benchmarks/run_benchmark.py, configs/*.yaml
  notebooks/
  tests/
  ```
- `pyproject.toml` (uv/hatch), pin `transformers>=5,<6`, `gradio>=6,<7`, `torch>=2.4`; `uv.lock`.
- `ruff` + `pytest` + GitHub Actions (lint, tests on CPU with a tiny randomly-initialised 2-layer ViT config so CI stays <2 min).
- Tests: rollout vs hand-computed 3×3 example and vs jacobgil reference; token layout for every registry entry; shape/NaN checks; metric monotonicity on a synthetic map.
- `LICENSE` (MIT), `CITATION.cff`, `CHANGELOG.md`, README rewrite (what it is, 30-second demo GIF, install, API, results table, citation), `CONTRIBUTING.md`.
- Rename consistently: repo **ViT-Insight**, package `vit_insight`, app title "ViT-Insight".

---

## 6. Timeline (today = Wed 2026-10-07; most US PhD deadlines Dec 1–15, MS Dec 15–Jan 15)

| Week | Dates | Milestone | Exit criterion |
|---|---|---|---|
| 1 | Oct 7–13 | Package restructure, C1–C8 fixed, tests + CI green, LICENSE | `pip install -e .` works; `pytest` passes; rollout matches reference |
| 2 | Oct 14–20 | Registry for 8 models, token layout, eager/caching/device, CLI | every model in §3.1 produces a map from the CLI |
| 3 | Oct 21–27 | Methods: corrected rollout variants, grad-rollout, Chefer-generic (incl. CLIP text), AttnLRP via lxt, baselines | all methods run on 10 images, maps look sane |
| 4 | Oct 28–Nov 3 | Metrics module, data loaders (ImageNet-1k subset, VOC), 300-image pilot | pilot tables for H1; pipeline timing known |
| 5 | Nov 4–10 | Full benchmark run (overnight / Colab), result tables + plots | CSV of all metrics × models × methods |
| 6 | Nov 11–17 | Explorer P1 features (CLIP label-conditional, comparison view, head grid, artifact inspector), HF Space live | public demo URL |
| 7 | Nov 18–24 | Write report (6–8 pp, CVPR-workshop style): intro, related work, method, results, limitations; figures from §5 | full draft |
| 8 | Nov 25–Dec 1 | Polish, README results section, arXiv submission (cs.CV, cross-list cs.LG) | arXiv id in hand for applications |
| 9–10 | Dec 2–15 | Buffer; sanity-check experiments; optional test-time registers (P2) | — |
| Feb–Mar 2027 | — | Submit to ICLR 2027 workshop (deadlines typically early Feb) or CVPR 2027 XAI4CV (typically mid-March; 2026 edition had 8-pp proceedings + 4-pp non-proceedings tracks) | — |

Effort is not the constraint (confirmed 2026-10-08); the binding constraint is the Dec 1 arXiv date. Protect weeks 7–8 for writing even if §3.6 additions are incomplete — they can land in v2 of the arXiv paper.

---

## 7. Deliverables for the application

1. **GitHub repo** with CI badge, tests, results table, demo GIF, arXiv badge.
2. **Hugging Face Space** — one link, loads in <10 s.
3. **arXiv technical report** (6–8 pages) — single-author or with a mentor if you can get one to read a draft.
4. **One-paragraph SoP blurb** (draft now, refine with results):
   > I built ViT-Insight, an open-source toolkit and controlled study of attention-based explanations in Vision Transformers. Holding architecture fixed, I measured how pretraining objective (supervised, CLIP, DINOv2) and register tokens change the faithfulness and localisation of explanations across N methods on ImageNet and VOC, finding that … The work is on arXiv and the interactive explorer has been used by …
5. **CV line**: "ViT-Insight — open-source ViT interpretability toolkit + benchmark (arXiv:XXXX.XXXXX, 2026)".

---

## 8. Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Transformers 5.x API drift (processor/attention kwargs) | Med | pin `<6`; CI catches it; keep eager attention explicit |
| ImageNet gated access or download size | Med | 1,000-image subset is ~130 MB; fall back to VOC-only + Imagenette if blocked |
| Compute on Mac too slow for full grid | Med | pilot first; Colab Pro (~$10) for the overnight run; reduce to 500 images |
| Results are "boring" (no effect of registers) | Low–Med | a clean null result with a correct pipeline is still publishable at a workshop and still a strong application artifact; frame as "when does attention suffice?" |
| AttnLRP/lxt doesn't support HF ViT 5.x out of the box | Med | lxt ships ViT examples; else drop to Chefer-generic as the only relevance method |
| Scope creep into SAEs / mech-interp | High (it's tempting) | explicitly future work; cite Prisma |

---

## 9. Decisions (confirmed 2026-10-08)

| Question | Decision | Effect on plan |
|---|---|---|
| Programme | **US PhD** applications, first deadlines Dec 1–15, 2026 | arXiv by Dec 1 stays the hard target; report must read as a research paper, not a tool paper |
| Time | Ample; "keep it rolling" | Full scope: keep SigLIP, AttnLRP, Attention Flow; add the ViT-L scale check (§3.6) |
| Compute | **Colab Pro with compute units (A100/L4) + MacBook M5** | Benchmark on 5,000 ImageNet-val images (5/class) instead of 1,000; test-time registers promoted from P2 to **P1**; dev + tests on the M5 (MPS), heavy runs on Colab |
| Venue | **arXiv is enough for now** | Workshop submission remains optional stretch (Feb–Mar 2027) |
| Research option | **A confirmed** (objective × registers faithfulness study) | §2.2 hypotheses H1–H5 are the thesis |
| Reader | None yet; may recruit one | Week 7: send draft to one external reader (former professor / senior colleague); 1 week turnaround budgeted in weeks 9–10 |

### 3.6 Scope additions enabled by the GPU budget

- **Scale check:** add `openai/clip-vit-large-patch14` and `google/vit-large-patch16-224` (24 layers, 16 heads) for H2/H3 only, to show findings are not B/16-specific.
- **Test-time registers (Jiang et al. 2025)** on CLIP-B/16 and DINOv2-B: the training-free counterpart to the trained-register pair; gives H3 a second, independent manipulation.
- **5,000-image faithfulness set** (5 per class) and full VOC val for localisation; report mean ± 95% CI via bootstrap over images.
- **Two perturbation baselines** (blur, dataset-mean) for deletion/insertion, since the Aug-2026 benchmark shows baseline choice flips rankings.

## 10. Sources

- Abnar & Zuidema, *Quantifying Attention Flow in Transformers*, ACL 2020 — https://arxiv.org/abs/2005.00928
- jacobgil/vit-explain reference rollout (identity term, `a @ result` order, head fusion, discard ratio) — https://github.com/jacobgil/vit-explain/blob/main/vit_rollout.py
- Chefer, Gur, Wolf, *Generic Attention-model Explainability for Interpreting Bi-Modal and Encoder-Decoder Transformers*, ICCV 2021 — https://arxiv.org/abs/2103.15679 · code https://github.com/hila-chefer/Transformer-MM-Explainability
- Achtibat et al., *AttnLRP*, ICML 2024 — https://github.com/rachtibat/LRP-eXplains-Transformers · docs https://lxt.readthedocs.io/
- Jiang, Dravid et al., *Vision Transformers Don't Need Trained Registers*, NeurIPS 2025 Spotlight — https://arxiv.org/abs/2506.08010 · code https://github.com/nickjiang2378/test-time-registers
- Gandelsman, Efros, Steinhardt, *Interpreting CLIP's Image Representation via Text-Based Decomposition*, ICLR 2024 — https://yossigandelsman.github.io/clip_decomposition/
- Nishankar et al., *Does Explainability Transfer? A Controlled Benchmark of Attribution Methods on Vision Transformers and CNNs*, Aug 2026 — https://arxiv.org/abs/2608.02396
- Jo, Jang, Park, *Decision-Aware Attention Propagation for ViT Explainability*, Apr 2026 — https://arxiv.org/abs/2604.18094
- *HiLRP: Conservation-Valid Attribution via Attention Primitives*, Sep 2026 — https://arxiv.org/abs/2609.01282
- Wu et al., *On the Faithfulness of Vision Transformer Explanations*, CVPR 2024 — https://openaccess.thecvf.com/content/CVPR2024/papers/Wu_On_the_Faithfulness_of_Vision_Transformer_Explanations_CVPR_2024_paper.pdf
- Zhang, *Sparse but not Simpler: A Multi-Level Interpretability Analysis of ViTs*, Mar 2026 — https://arxiv.org/abs/2603.15919
- Joseph et al., *Prisma: An Open Source Toolkit for Mechanistic Interpretability in Vision and Video*, 2025 — https://github.com/prisma-multimodal/vit-prisma
- *Explainability of Vision Transformers: a comprehensive review and new perspectives* (TMLR submission, 2026) — https://openreview.net/pdf/38dbba3b7e01ab08fe4d9a97a11fd651c8627579.pdf
- Quantus toolkit (0.6.0) — https://pypi.org/project/quantus/
- HF transformers: `output_attentions` requires eager attention — https://github.com/huggingface/transformers/issues/30978 · https://github.com/huggingface/transformers/issues/41929
- XAI4CV @ CVPR 2026 (format reference for 2027) — https://xai4cv-workshop.github.io/xai4cv2026/
- ICLR 2027 — https://iclr.cc/Conferences/2027/CallForPapers · TMLR (rolling) — https://jmlr.org/tmlr/
