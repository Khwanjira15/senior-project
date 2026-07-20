# Streamlit Community Cloud Deploy Guide

## Files to use

- Main app file: `streamlit_app.py`
- Python dependencies: `requirements.txt`

## Deploy steps

1. Push this project to a GitHub repository.
2. Sign in to Streamlit Community Cloud with GitHub.
3. Click **New app**.
4. Choose the repository, branch, and set:
   - Main file path: `streamlit_app.py`
5. Click **Deploy**.

## Notes

- This app runs PyTorch inference on CPU by default, so the first prediction can be slow.
- HEIC and DICOM support depend on optional Python packages listed in `requirements.txt`.
- Avoid uploading patient-identifiable medical images to a public demo.
