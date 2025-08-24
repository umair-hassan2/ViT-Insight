import gradio as gr
from model_loader import load_model, run_inference
from inference import generate_attention_frames

processor, model = load_model()


def gradio_interface(user_image, labels, alpha=0.5, colormap="jet", mode="Per-layer GIF", layer_from=1, layer_to=None):
    # Require user-provided labels (1-5, comma-separated)
    if labels is None or str(labels).strip() == "":
        raise gr.Error("Please enter 1-5 labels (comma-separated).")
    text_labels = [s.strip() for s in str(labels).split(",") if s.strip()]
    if len(text_labels) == 0:
        raise gr.Error("Please enter 1-5 labels (comma-separated).")
    if len(text_labels) > 5:
        text_labels = text_labels[:5]

    img, attentions, pred = run_inference(model, processor, user_image, text_labels=text_labels)
    layer_range = (layer_from-1, layer_to) if layer_to is not None else None
    viz = generate_attention_frames(img, attentions, mode=mode, alpha=alpha, colormap=colormap, layer_range=layer_range)

    # Map predicted index to label string
    idx = int(pred)
    pred_label = text_labels[idx] if 0 <= idx < len(text_labels) else str(idx)
    return viz, pred_label


iface = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.Image(type="pil", label="Upload Image"),
        gr.Textbox(label="Labels (comma-separated, max 5)", placeholder="e.g., a photo of a cat, a photo of a dog"),
        gr.Slider(0.0, 1.0, 0.5, label="Heatmap Alpha"),
        gr.Dropdown(["jet","viridis","hot"], value="jet", label="Colormap"),
        gr.Radio(["Per-layer GIF","Attention Rollout"], value="Per-layer GIF", label="Mode"),
        gr.Number(value=0, label="Layer From"),
        gr.Number(value=None, label="Layer To (Optional)")
    ],
    outputs=[
        gr.Image(type="filepath", label="Attention Visualization"),
        gr.Textbox(label="Predicted Label")
    ],
    title="ViT Attention Explorer",
    description="Upload an image and explore Vision Transformer per-layer attention maps or rollout."
)

iface.launch()
