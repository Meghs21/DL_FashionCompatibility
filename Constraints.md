# Constraints.md

## Technical Boundaries & Constraints

1. **Framework Boundary**:
   - Deep Learning code must remain compatible with standard PyTorch (`torch>=2.0.0`, `torchvision>=0.15.0`).
   - Do not replace PyTorch with alternative ML frameworks without explicit approval.

2. **API & Interface Compatibility**:
   - The primary prediction endpoint is `POST /predict`. It must accept multipart image uploads with keys: `top` (required), `bottom` (required), `footwear` (optional), `accessories` (optional).
   - Response payload must maintain backward compatibility with keys: `compatibility`, `color_harmony_score`, `color_swatches`, `feedback`, `feedback_details`, `styling_tips`, `item_breakdown`.

3. **Performance & Latency**:
   - Model inference must run in $< 1.5$ seconds per outfit on CPU environments.
   - Images are resized to $224 \times 224$ prior to tensor conversion to constrain memory usage.

4. **Encoding & Windows Console Safety**:
   - Terminal print statements inside `app.py` and Python scripts must use ASCII-safe characters (avoid unencoded unicode emojis in `print()`) to prevent `UnicodeEncodeError` on Windows `cp1252` stdout streams.

5. **Deployment Port**:
   - Environment variable `PORT` defaults to `5000`.
