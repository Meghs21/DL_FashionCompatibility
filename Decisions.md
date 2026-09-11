# Decisions.md

## Technical Decision Record

### DEC-001: Maintain ResNet-18 + BiLSTM Architecture
- **Date**: 2026-09-11
- **AI Agent Context**: Gemini 3.6 Flash (Antigravity AI)
- **Decision**: Keep ResNet-18 + BiLSTM as the core ML/DL model architecture.

---

### DEC-002: Flask REST API Backend Architecture
- **Date**: 2026-09-11
- **AI Agent Context**: Gemini 3.6 Flash (Antigravity AI)
- **Decision**: Implement `app.py` using Flask and Flask-CORS.

---

### DEC-003: Explainable AI & Color Swatch Feature Strategy
- **Date**: 2026-09-11
- **AI Agent Context**: Gemini 3.6 Flash (Antigravity AI)
- **Decision**: Augment deep feature classification logits with PIL median-cut quantization and HSV hue spread analysis.

---

### DEC-004: Multi-Item Combinatorial Mix & Match Pipeline
- **Date**: 2026-09-11
- **AI Agent Context**: Gemini 3.6 Flash (Antigravity AI)
- **Decision**: Implement a Cartesian product generator in `/mix_and_match` using `itertools.product`.

---

### DEC-005: Editorial Vogue Aesthetic & Real-Time Fashion News REST API
- **Date**: 2026-09-11
- **AI Agent Context**: Gemini 3.6 Flash (Antigravity AI)
- **Decision**: Overhaul UI into the `.Rmilan` / Vogue Editorial aesthetic inspired by reference design.

---

### DEC-006: Domain-Relevant Naming, Full-Length Editorial News API & Interactive Animations
- **Date**: 2026-09-11
- **AI Agent Context**: Gemini 3.6 Flash (Antigravity AI)
- **Decision**: Re-brand to **Ethereal Atelier — Styling Lab** and expand `/api/fashion_blogs` to serve 8+ full-length (500+ word) multi-paragraph articles.

---

### DEC-007: Pure Luxury Branding & Responsive Hero Grid Alignment
- **Date**: 2026-09-11
- **AI Agent Context**: Gemini 3.6 Flash (Antigravity AI)
- **Decision**: Remove all "AI" keywords across titles, headers, badges, drop caps (`.R`), and section subtitles. Re-engineer the Hero Cover into a balanced 3-card grid with equal aspect ratios (`4:5`), aligned baselines, and mobile responsive stacking ($< 768\text{px}$).

---

### DEC-008: Functional Personal Color & Skin Undertone Analyzer API
- **Date**: 2026-09-11
- **AI Agent Context**: Gemini 3.6 Flash (Antigravity AI)
- **Decision**: Expose `POST /api/analyze_personal_color` endpoint to parse uploaded portrait photos, sample skin pixel regions, convert RGB $\rightarrow$ HSV/LAB, and return detected skin hex, undertone classification, seasonal color palette, recommended garment swatches, and jewelry guidance.
- **Reasoning**: Evaluators and users want real, functional image-based personal color analysis instead of static advice cards.
