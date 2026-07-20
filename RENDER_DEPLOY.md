# Render Deploy Guide

## What is included

- `requirements.txt` for Render build dependencies
- `render.yaml` with build/start commands and health check
- `webapp.py` production-friendly host/port binding
- `/healthz` endpoint for Render health checks
- Linux-friendly optional medical image support via `pydicom` and `pillow-heif`

## Deploy steps

1. Push this project to GitHub.
2. Create a new Render Web Service from the repository.
3. Render should detect `render.yaml` automatically.
4. Wait for the first build to finish.
5. Open the generated public URL and test the upload flow.

## Important notes

- This app runs PyTorch inference on CPU by default, so the first prediction can be slow.
- DICOM support on Render depends on `pydicom` and whether the uploaded file uses a compatible transfer syntax.
- If you expect many users, move inference to a GPU-capable service or add a job queue.
- Make sure no real patient-identifiable images are exposed in a public deployment.
