# Rollback.md

## Rollback Procedures

### Reverting Web Application Changes
If it is ever necessary to roll back to the original CLI test scripts prior to the Web Studio overhaul:

1. **Git Commit Revert**:
   ```bash
   git checkout main
   git reset --hard HEAD~1
   ```
2. **Files Affected by Rollback**:
   - `app.py` (Deletes REST server)
   - `Dockerfile` & `render.yaml` (Deletes deployment scripts)
   - `requirements.txt` (Deletes web dependencies)
   - `index.html`, `styles.css`, `script.js` (Restores legacy HTML interface)
3. **Verification After Rollback**:
   - Verify CLI inference runs via:
     ```bash
     python test_custom_compatibility.py --images <image1.jpg> <image2.jpg>
     ```
