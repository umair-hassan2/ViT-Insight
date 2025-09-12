
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

### Notation

* ![eq_a](https://quicklatex.com/cache3/c7/ql_36269e6e845c142142517c24f24383c7_l3.png) = attention matrix at layer *l* (per head)  
* ![eq_b](https://quicklatex.com/cache3/ee/ql_704108d56a49b7ae3c1cd1c7aaf203ee_l3.png) = average attention across heads at layer *l*  
* ![eq_c](https://quicklatex.com/cache3/fc/ql_838468c12ff5fd3709898d1d50ff67fc_l3.png) = attention matrix after adding residuals  
* ![eq_d](https://quicklatex.com/cache3/65/ql_2bf949f7a1b1cc1e98ba112287191265_l3.png) = rollout matrix up to layer *l*  
* ![eq_e](https://quicklatex.com/cache3/02/ql_b80b96d2c3f785e26e8688a0968e9402_l3.png) = identity matrix  
* ![eq_f](https://quicklatex.com/cache3/25/ql_f212e665f290020b3d06b5a7b6eea825_l3.png) = residual scaling factor (usually 1)

---

### Step 1: Average attention across heads
![eq_1](https://quicklatex.com/cache3/c8/ql_7a185ae6a6394d5b525d48af33bbc4c8_l3.png)

### Step 2: Add identity (residual connection effect)

![eq_2](https://quicklatex.com/cache3/34/ql_97a71a469aa24769204766bcf0ef3534_l3.png)

### Step 3: Recursive rollout definition

![eq_3](https://quicklatex.com/cache3/c7/ql_d88420789066fdbf504dd1d898883ec7_l3.png)

### Final rollout (after L layers)

![eq_4](https://quicklatex.com/cache3/58/ql_9f971915eb569ee2105c0369e9aace58_l3.png)

* ![eq_5](https://quicklatex.com/cache3/8b/ql_53e973516ffc6cddef0694854f00c78b_l3.png) highlights **which input patches influence the `[CLS]` token** the most.

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
