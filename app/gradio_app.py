"""Gradio front-end for ViT-Insight. Run: python app/gradio_app.py"""

from __future__ import annotations

import tempfile

import gradio as gr

from vit_insight import Registry, load_model, run_forward
from vit_insight.viz import head_grid, per_layer_frames, rollout_overlay, save_gif

registry = Registry()
MODEL_IDS = [s.id for s in registry if s.cls_attention]
COLORMAPS = ["jet", "viridis", "hot", "magma"]


def _labels(text: str) -> list[str]:
    labels = [s.strip() for s in (text or "").split(",") if s.strip()]
    if not labels:
        raise gr.Error("Enter 1-5 comma-separated labels for this model.")
    return labels[:5]


def explain(
    model_id,
    image,
    labels,
    mode,
    head_fusion,
    discard_ratio,
    residual,
    alpha,
    cmap,
    layer_from,
    layer_to,
    head_layer,
):
    if not model_id:
        raise gr.Error("Select a model.")
    if image is None:
        raise gr.Error("Upload an image.")
    lm = load_model(model_id)
    spec = lm.spec
    text_labels = _labels(labels) if spec.text_conditioned else None
    res = run_forward(lm, image, text_labels)

    n = len(res.attentions)
    lo = max(0, int(layer_from) - 1)
    hi = min(n, int(layer_to)) if layer_to else n
    if lo >= hi:
        raise gr.Error(f"Layer range must satisfy 1 <= from < to <= {n}.")

    if mode == "Attention Rollout":
        out = rollout_overlay(
            res,
            spec,
            head_fusion=head_fusion,
            discard_ratio=discard_ratio,
            residual=residual,
            start_layer=lo,
            end_layer=hi,
            alpha=alpha,
            cmap=cmap,
        )
    elif mode == "Per-layer GIF":
        frames = per_layer_frames(res, spec, range(lo, hi), head_fusion, alpha, cmap)
        out = save_gif(frames, tempfile.NamedTemporaryFile(delete=False, suffix=".gif").name)
    else:  # Head grid
        out = head_grid(res, spec, int(head_layer) - 1, alpha, cmap)

    if res.pred is None:
        pred = "(encoder has no classification head)"
    elif text_labels:
        pred = f"{text_labels[res.pred]}  (p={res.pred_prob:.3f})"
    else:
        pred = f"{lm.model.config.id2label.get(res.pred, res.pred)}  (p={res.pred_prob:.3f})"
    return out, pred


def on_model_change(model_id):
    if not model_id:
        return gr.update(), gr.update()
    spec = registry.get(model_id)
    return gr.update(visible=spec.text_conditioned), gr.update(value=spec.num_layers)


with gr.Blocks(title="ViT-Insight") as demo:
    gr.Markdown(
        "## ViT-Insight\n"
        "Attention rollout, per-layer and per-head attention maps for Vision Transformers."
    )
    with gr.Row():
        model_dd = gr.Dropdown(MODEL_IDS, label="Model")
        mode = gr.Radio(
            ["Attention Rollout", "Per-layer GIF", "Head grid"],
            value="Attention Rollout",
            label="View",
        )
    with gr.Row():
        image = gr.Image(type="pil", label="Image")
        with gr.Column():
            labels = gr.Textbox(
                label="Text labels (comma-separated, max 5)",
                placeholder="a photo of a cat, a photo of a dog",
                visible=False,
            )
            pred_out = gr.Textbox(label="Prediction")
    with gr.Accordion("Rollout options", open=False):
        with gr.Row():
            head_fusion = gr.Dropdown(["mean", "max", "min"], value="mean", label="Head fusion")
            discard_ratio = gr.Slider(0.0, 0.95, 0.0, step=0.05, label="Discard ratio")
            residual = gr.Checkbox(True, label="Residual term (½A + ½I), per Abnar & Zuidema")
    with gr.Row():
        alpha = gr.Slider(0.0, 1.0, 0.5, label="Heatmap alpha")
        cmap = gr.Dropdown(COLORMAPS, value="jet", label="Colormap")
        layer_from = gr.Number(1, label="Layer from", precision=0)
        layer_to = gr.Number(12, label="Layer to", precision=0)
        head_layer = gr.Number(12, label="Layer (head grid)", precision=0)
    run = gr.Button("Explain", variant="primary")
    viz_out = gr.Image(label="Visualization")

    model_dd.change(on_model_change, model_dd, [labels, layer_to])
    run.click(
        explain,
        [
            model_dd,
            image,
            labels,
            mode,
            head_fusion,
            discard_ratio,
            residual,
            alpha,
            cmap,
            layer_from,
            layer_to,
            head_layer,
        ],
        [viz_out, pred_out],
    )

if __name__ == "__main__":
    demo.launch()
