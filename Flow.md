# Flow.md

## Data & Execution Flow Traces

### Flow 1: Single Compatibility Prediction & Explainable Feedback Pipeline
```
[ HTTP Multipart Request ] ──► Image Ingestion ──► Tensor Prep ──► PyTorch ResNet+BiLSTM ──► JSON Output
```

### Flow 2: Multi-Item Combinatorial Mix & Match Pipeline
```
[ Wardrobe Item Pools ] ──► itertools.product() ──► PyTorch Batch Eval ──► Score Ranking ──► Comparative AI Notes
```

### Flow 3: Real-Time Vogue Fashion News & Article Reader Pipeline
```
[ Frontend Initialization / Category Filter Click ]
       │
       ▼
[ app.py: GET /api/fashion_blogs ]
       │
       ├─► Returns JSON list of real-time editorial trend articles
       │     Fields: id, title, subtitle, category, author, read_time, date, image, summary, content
       │
       ▼
[ script.js: renderBlogs() ]
       │
       ├─► Renders Vogue editorial cards with image zoom hover effects
       └─► On Card Click: Triggers openBlogModal(id) -> Displays full modal article reader
```
