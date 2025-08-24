from transformers import CLIPProcessor, CLIPModel
import torch

MODEL_ID = "openai/clip-vit-base-patch32"

def load_model(model_id=MODEL_ID):
    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id)
    model.eval()
    return processor, model

def run_inference(model, processor, image, text_labels):
    inputs = processor(text=text_labels, images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    logits_per_image = outputs.logits_per_image
    probs = torch.softmax(logits_per_image, dim=1)
    pred = torch.argmax(probs, dim=1).item()
    img_tensor = inputs.pixel_values[0]
    img = (img_tensor.permute(1,2,0).cpu().numpy() - img_tensor.min().cpu().numpy()) / (img_tensor.max().cpu().numpy() - img_tensor.min().cpu().numpy())
    attentions = outputs.vision_model_output.attentions
    return img, attentions, pred
