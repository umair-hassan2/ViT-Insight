from transformers import CLIPProcessor, CLIPModel
import torch

MODEL_ID = "openai/clip-vit-base-patch32"
VISION_BASE = "vision-base"

def load_model(model_id=MODEL_ID):
    # Deprecated: prefer using ModelManager.load
    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id)
    model.eval()
    return processor, model

# run_inference moved to ModelManager.run_inference
