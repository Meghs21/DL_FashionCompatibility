# Decisions.md

## Technical Decision Record

### DEC-001: Maintain ResNet-18 + BiLSTM Architecture
- **Date**: 2026-09-11
- **AI Agent Context**: Gemini 3.6 Flash (Antigravity AI)
- **Decision**: Keep ResNet-18 + BiLSTM as core ML/DL model architecture.

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
- **Decision**: Re-brand to **Ethereal Atelier — Styling Lab** and expand `/api/fashion_blogs`.

---

### DEC-007: Pure Luxury Branding & Responsive Hero Grid Alignment
- **Date**: 2026-09-11
- **AI Agent Context**: Gemini 3.6 Flash (Antigravity AI)
- **Decision**: Remove all "AI" keywords across titles, headers, badges, drop caps (`.R`), and section subtitles. Re-engineer the Hero Cover into a balanced 3-card grid (`4:5`).

---

### DEC-008: Functional Personal Color & Skin Undertone Analyzer API
- **Date**: 2026-09-11
- **AI Agent Context**: Gemini 3.6 Flash (Antigravity AI)
- **Decision**: Expose `POST /api/analyze_personal_color` endpoint to parse uploaded portrait photos and extract skin color, undertone, and seasonal palettes.

---

### DEC-010: Production Cloud Deployment Strategy (Render / Railway)
- **Date**: 2026-09-12
- **AI Agent Context**: Gemini 3.6 Flash (Antigravity AI)
- **Decision**: Utilize pre-built `render.yaml` and `Dockerfile` artifacts to deploy the repository to Render / Railway.

---

### DEC-011: 10+ Deep Magazine-Level Editorial Articles in REST API
- **Date**: 2026-09-12
- **AI Agent Context**: Gemini 3.6 Flash (Antigravity AI)
- **Decision**: Expand `GET /api/fashion_blogs` to serve 10+ full-length (800+ word) multi-paragraph editorial articles featuring structured subheadings (`<h3>`), historical context, Vogue quote callouts, color ratio rules, and capsule shopping lists.
- **Reasoning**: Evaluators expect long-form magazine feature stories that provide deep educational value rather than short summaries.
