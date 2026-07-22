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

## Optional external checkpoints

If a checkpoint is too large for GitHub, the app can load it from an external downloadable URL instead.

Set an environment variable in Streamlit Community Cloud:

- `CHECKPOINT_URL_RESNET50_BAM`
- `CHECKPOINT_URL_RESNET50_COORDATT`

The value should be a direct `.pth` download URL.

Examples:

- `CHECKPOINT_URL_RESNET50_BAM=https://.../ResNet50_final.pth`
- `CHECKPOINT_URL_RESNET50_COORDATT=https://.../ResNet50_final.pth`

The app will:

1. Look for the local checkpoint in the repository
2. If missing, look for the matching `CHECKPOINT_URL_...`
3. Download the file on first use and cache it temporarily on the server
