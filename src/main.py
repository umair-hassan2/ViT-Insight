import gradio as gr
from model_config import ModelConfig
from model_manager import ModelManager
from inference import generate_attention_frames

config = ModelConfig()
manager = ModelManager(config)

def is_text_based(model_id: str) -> bool:
    return config.get_model_type(model_id) == "text-based"

def gradio_interface(model_id, user_image, labels, alpha=0.5, colormap="jet",
                     mode="Per-layer GIF", layer_from=1, layer_to=None):
    if model_id is None:
        raise gr.Error("Select a model.")
    processor, model = manager.load(model_id)  # assume internal caching
    num_layers = manager.get_num_layers(model_id=model_id)
    text_based = is_text_based(model_id)

    text_labels = []
    if text_based:
        if labels is None or str(labels).strip() == "":
            raise gr.Error("Please enter 1-5 labels (comma-separated).")
        text_labels = [s.strip() for s in str(labels).split(",") if s.strip()]
        if len(text_labels) == 0:
            raise gr.Error("Please enter 1-5 labels (comma-separated).")
        if len(text_labels) > 5:
            text_labels = text_labels[:5]

    img, attentions, pred = manager.run_inference(
        model_id, model, processor, user_image,
        text_labels=text_labels if text_based else None
    )

    if layer_to is not None and layer_to > num_layers:
        raise gr.Error(f"Layer To cannot exceed the model's number of layers ({num_layers}).")

    layer_range = (layer_from - 1, layer_to) if layer_to is not None else None
    viz = generate_attention_frames(
        img, attentions, mode=mode, alpha=alpha,
        colormap=colormap, layer_range=layer_range
    )

    pred_label = ""
    if text_based:
        try:
            idx = int(pred)
            pred_label = text_labels[idx] if 0 <= idx < len(text_labels) else str(idx)
        except Exception:
            pred_label = str(pred)
    return viz, pred_label

def on_model_change(model_id):
    show = is_text_based(model_id) if model_id else False
    num_layers = manager.get_num_layers(model_id) if model_id else None
    return (
        gr.update(visible=show),  # labels textbox
        gr.update(visible=show),   # predicted label output
        gr.update(value=num_layers)  # layer_to default
    )

model_ids = config.get_model_ids()

with gr.Blocks(title="ViT Attention Explorer") as demo:
    gr.Markdown("## ViT Attention Explorer\nUpload an image and explore Vision Transformer per-layer attention maps or rollout.")
    with gr.Row():
        model_dd = gr.Dropdown(choices=model_ids, value=None, label="Model ID")
        colormap = gr.Dropdown(["jet", "viridis", "hot"], value="jet", label="Colormap")
        mode = gr.Radio(["Per-layer GIF", "Attention Rollout"], value="Per-layer GIF", label="Mode")

    with gr.Row():
        image = gr.Image(type="pil", label="Upload Image")
        labels = gr.Textbox(
            label="Labels (comma-separated, max 5)",
            placeholder="e.g., a photo of a cat, a photo of a dog",
            visible=False
        )

    with gr.Row():
        alpha = gr.Slider(0.0, 1.0, 0.5, label="Heatmap Alpha")
        layer_from = gr.Number(value=1, label="Layer From")
        layer_to = gr.Number(value=None, label="Layer To (Optional)")

    run_btn = gr.Button("Run Inference")

    with gr.Row():
        viz_out = gr.Image(type="filepath", label="Attention Visualization")
        pred_out = gr.Textbox(label="Predicted Label", visible=False)

    model_dd.change(on_model_change, inputs=model_dd, outputs=[labels, pred_out, layer_to])

    run_btn.click(
        gradio_interface,
        inputs=[model_dd, image, labels, alpha, colormap, mode, layer_from, layer_to],
        outputs=[viz_out, pred_out]
    )

demo.launch()
