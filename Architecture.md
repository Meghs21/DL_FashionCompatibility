# Architecture.md

## System Overview
**Ethereal Atelier (.Rmilan)** is a full-stack high-fashion editorial magazine platform and AI compatibility analysis engine:

1. **Vogue Editorial Frontend** (`index.html`, `styles.css`, `script.js`): High-fashion magazine platform inspired by the `.Rmilan` reference design:
   - Deep Forest Emerald (`#08332b`) and Sunflower Yellow (`#facc15`) palette.
   - Ultra-bold `FAS. STYLINGOX` header with Playfair drop-caps (`F`, `R`, `.Rmilan`) and directional arrows (`↗`).
   - Real-Time Vogue Fashion News & Trends tab with article reader modals.
   - Multi-item wardrobe pool uploader & combinatorial ranking lookbook.
2. **REST API Service** (`app.py`): Flask backend handling:
   - `GET /api/fashion_blogs`: Serves real-time editorial trend feeds, Paris/Milan runway news, Pantone color reports, and styling masterclasses.
   - `POST /predict`: Single outfit compatibility.
   - `POST /mix_and_match`: Multi-item wardrobe pool combinatorial processing, batch PyTorch model inference, score ranking, and comparative AI reasoning synthesis.
3. **Deep Learning Model Engine** (`models/`):
   - `ResNetEncoder` (`models/resnet_encoder.py` & `models/resnet18_model.py`): 512-dim visual feature extraction per garment.
   - `OutfitLSTM` (`models/lstm_model.py`): BiLSTM sequence compatibility score classifier.

```
┌────────────────────────────────────────────────────────────────────────┐
│                   .Rmilan Vogue Editorial Platform                     │
│    (FAS. STYLINGOX Cover | Real-Time News Feed | Wardrobe Studio)      │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ REST API & Multipart Requests
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                         Flask REST Backend API                         │
│       (GET /api/fashion_blogs | POST /predict | POST /mix_and_match)   │
└──────────────┬────────────────────┬────────────────────┬───────────────┘
               │                    │                    │
               ▼                    ▼                    ▼
┌──────────────────────────┐ ┌────────────────────┐ ┌───────────────────┐
│ Real-time News Feed Engine│ │ PyTorch Model      │ │ Color & Harmony   │
│ (Vogue Editorial Trends) │ │ ResNet18 + BiLSTM  │ │ HSV Palette Engine│
└──────────────────────────┘ └────────────────────┘ └───────────────────┘
```
