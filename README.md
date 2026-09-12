# 👗 Ethereal Atelier — AI Fashion Compatibility Engine

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0+-green.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

**Ethereal Atelier** is a Deep Learning powered Fashion Compatibility Analysis and Recommendation Engine. It evaluates whether clothing items form a stylish, harmonized outfit using computer vision and sequence modeling techniques.

---

## 🏗️ System Architecture & Workflow

```
┌─────────────────┐       Multipart HTTP       ┌────────────────────────┐
│   Web Studio    │ ─────────────────────────> │   Flask REST Backend   │
│ (Glassmorphism) │ <───────────────────────── │       (app.py)         │
└─────────────────┘      JSON Prediction       └───────────┬────────────┘
                                                           │
                                             ┌─────────────┴─────────────┐
                                             ▼                           ▼
                                ┌─────────────────────────┐ ┌─────────────────────────┐
                                │  PyTorch DL Model       │ │  Color & Style Analyzer │
                                │  • ResNet-18 Encoder    │ │  • Dominant Hex Swatches│
                                │  • BiLSTM Aggregator    │ │  • Hue Variance Harmony │
                                └─────────────────────────┘ └─────────────────────────┘
```

1. **Feature Extraction (ResNet-18)**: Pre-trained CNN extracts 512-dimensional visual feature vectors for each clothing item picture (top, bottom, footwear, accessory).
2. **Sequence Modeling (BiLSTM)**: A Bidirectional Long Short-Term Memory network captures pairwise and global relationships across the outfit sequence.
3. **Color & Palette Analysis**: Extracts dominant hex color palettes and calculates color wheel variance to assess visual harmony.
4. **Explainable AI (XAI) Stylist**: Translates deep feature logits into readable feedback, color swatches, and styling recommendations.

---

## ⚡ Quick Start & Setup

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/Meghs21/DL_FashionCompatibility.git
cd DL_FashionCompatibility
pip install -r requirements.txt
```

### 2. Run the Application
```bash
python app.py
```
Open your browser at `http://127.0.0.1:5000` to launch the **AI Fashion Compatibility Studio**.

### 3. Quick Presets for Demo / Presentation
Click any preset button (**Casual Chic**, **Executive Formal**, **Urban Streetwear**) in the UI to instantly load sample outfit items and evaluate compatibility in 1-click!

---

## 🐳 Production Deployment

### Option A: Docker Container
```bash
docker build -t fashion-compatibility-ai .
docker run -p 5000:5000 fashion-compatibility-ai
```

### Option B: Deploy to Render / Railway / Cloud
- Connect this repository to **Render** or **Railway**.
- Set build command to: `pip install -r requirements.txt`
- Set start command to: `gunicorn app:app`
- Render configuration is pre-configured in `render.yaml`.

---

## 📊 Dataset & Model Details

- **Dataset**: Polyvore Outfits (Disjoint split).
- **Loss Function**: `BCEWithLogitsLoss` (Binary Cross-Entropy over logits).
- **Optimization**: Adam optimizer with Mixed Precision Training (`torch.amp.autocast`).
- **Evaluation Metric**: AUC-ROC & Binary Classification Accuracy.

---



* **Q: Why combine ResNet-18 with BiLSTM instead of a standard classifier?**
  * *Answer*: Clothing items in an outfit are inherently sequential and context-dependent. ResNet-18 acts as the spatial feature extractor, while BiLSTM models how items relate to each other sequentially (e.g. how footwear complements trousers, which complement tops).
* **Q: Why choose ResNet-18 over a Vision Transformer (ViT)?**
  * *Answer*: ResNet-18 offers lower parameter overhead, lower memory footprint, and faster inference latency (crucial for real-time web deployment) while preserving high accuracy on Polyvore benchmark datasets.
* **Q: How does the system handle variable outfit lengths?**
  * *Answer*: We use `torch.nn.utils.rnn.pack_padded_sequence` to dynamic batch outfits containing 2, 3, or 4 items without wasting computation on zero-padded slots.
* **Q: How is explainability achieved?**
  * *Answer*: Beyond binary classification, our custom API extracts dominant color swatches, evaluates hue variance, and provides structured stylist feedback notes.
