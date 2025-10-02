
# Vision Attention Explorer

This project visualizes **attention maps** and computes **Attention Rollout** for Vision Transformer (ViT) models.
It helps interpret **how information flows across layers, heads, and tokens** (especially `[CLS]`) and allows interactive exploration per image, per label, and per layer.

---

## 🔍 Features

* **Per-layer attention visualization:** view heatmaps for individual transformer layers.
* **Multi-label support:** select multiple text labels and see which patches align most strongly with each label.
* **Average attention across heads:** attention is averaged across all heads for interpretability.
* **Customizable heatmap overlays:** adjust `alpha` transparency and select colormap for visual clarity.
* **Layer range selection:** choose which layers to view from `start` → `end`.
* **Attention Rollout:** trace cumulative influence of patches on `[CLS]` token across layers.
* **GIF generation:** create dynamic visualizations across layers.

---

## 🔍 Attention Rollout

The rollout progressively multiplies attention matrices across layers to trace how much each input patch contributes to the final `[CLS]` decision.

<img width="646" height="641" alt="Screenshot 2025-10-02 at 11 07 43 PM" src="https://github.com/user-attachments/assets/8c5c1a60-408d-46d1-8031-5fd110c7e286" />


---

## 📊 Layer-wise Visualization

1. **Patch embeddings per layer:**  
   Extract per-layer patch embeddings from ViT and normalize them.  
   ![eq_6](https://quicklatex.com/cache3/70/ql_7e1d57325bb83c75bbf237256a752370_l3.png)

2. **Overlay heatmaps:**
   * Red = strong match to label / high attention  
   * Blue = weak match  
   * Adjustable alpha and colormap for better visualization

3. **Dynamic GIF:**
   * Shows attention evolution across layers  
   * Optional per-layer snapshots for deep inspection

4. **Layer range selection:**  
   Allows users to focus on a subset of layers for analysis instead of all layers.

---

## 🖼️ Example Outputs

<p align="center">
  <b>Per-layer Attention Evolution (GIF)</b><br>
  <img src="https://github.com/user-attachments/assets/caaa7360-f4ff-403b-b0ce-a74508307d2f" width="400">
</p>

<p align="center">
  <b>Attention Rollout Map</b><br>
  <img src="https://github.com/user-attachments/assets/d1816147-3b4f-4ce9-b6ee-e13beb436bce" width="400">
</p>

---

## 💡 Use Cases

* Visualize **object-level focus** of ViTs on an image.
* Compare attention across **different labels** or **image classes**.
* Explain **model decisions** for research, portfolios, or grad applications.
* Understand **layer-wise behavior** of transformer-based vision models.

---

## 🚀 Getting Started

Follow these steps to run the Vision Attention Explorer locally:

1. **Clone the repository**

```bash
git clone https://github.com/your-username/vision-attention-explorer.git
cd vision-attention-explorer
```

2. **Install dependencies**
```
pip install -r requirements.txt
```
3. **Run the application**
```
python3 src/main.py
```
4. Access the app
```
http://127.0.0.1:7860
```

---
## 📚 Reference

The method was proposed in:
*Abnar, S., & Zuidema, W. (2020). "Quantifying Attention Flow in Transformers." ACL 2020.*
[Paper Link](https://arxiv.org/abs/2005.00928)

---
