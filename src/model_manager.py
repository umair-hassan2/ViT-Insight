from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional
from transformers import(
        CLIPProcessor,
        CLIPModel,
        ViTImageProcessor,
        ViTForImageClassification,
        AutoProcessor,
        AutoModel,
        AutoModelForImageClassification,
        AutoConfig
)
import torch
import model_config

class ModelManager:
    DEFAULT_LAYERS= 12

    def __init__(self, base_config: model_config.ModelConfig):
        self._base_config = base_config

    def list_models(self) -> List[Dict[str, str]]:
        models = self._base_config.get_model_ids()
        return models

    def load(self, model_id: str, device: Optional[str] = None) -> Tuple[Any, Any]:
        processor_cls, model_cls = self._resolve_classes(model_id)
        processor = processor_cls.from_pretrained(model_id)
        model = model_cls.from_pretrained(model_id, output_attentions=True)
        model.eval()
        return processor, model

    def _resolve_classes(self, model_id: str):
        lower_id = model_id.lower()

        if "clip" in lower_id:
            return CLIPProcessor, CLIPModel
        if "vit" in lower_id:
            return ViTImageProcessor, ViTForImageClassification

        if any(k in lower_id for k in ["classif", "imagenet", "label"]):
            return AutoProcessor, AutoModelForImageClassification
        return AutoProcessor, AutoModel

    load_model = load

    def run_inference(self, model_id: str, model, processor, image, text_labels=None):
        """
        Runs a forward pass returning (img_np, attentions, pred_index).

        model_type sourcing:
            text-based  -> expects text labels (CLIP style)
            otherwise   -> pure vision classification (ViT)
        """
        model_type = self._base_config.get_model_type(model_id)

        if model_type == "text-based":
            if not text_labels:
                raise ValueError("text_labels required for text-based model.")
            inputs = processor(text=text_labels, images=image, return_tensors="pt", padding=True)
        else:
            inputs = processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)

        if model_type == "text-based":
            logits = outputs.logits_per_image  # (B, T)
        else:
            logits = outputs.logits  # (B, C)

        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()

        img_tensor = inputs.pixel_values[0]
        img = (img_tensor.permute(1, 2, 0).cpu().numpy() - img_tensor.min().cpu().numpy()) / (
            img_tensor.max().cpu().numpy() - img_tensor.min().cpu().numpy() + 1e-8
        )

        if model_type == "text-based":
            attentions = outputs.vision_model_output.attentions
        else:
            attentions = outputs.attentions

        return img, attentions, pred
    
    def get_num_layers(self, model_id) -> int:
        total_layers = self.DEFAULT_LAYERS
        try:
            total_layers = self._base_config.get_hidden_layers(model_id)
        except Exception as e:
            print(e)
        return total_layers
            
