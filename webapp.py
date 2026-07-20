"""Flask entrypoint for the model demo web app."""

from __future__ import annotations

import os

from flask import Flask, jsonify, render_template, request

from inference_service import ATTENTION_TYPES, MODEL_OPTIONS, predict_image, save_upload_to_temp


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024


@app.get("/")
def index():
    return render_template("index.html", model_options=MODEL_OPTIONS, attention_types=ATTENTION_TYPES)


@app.get("/healthz")
def healthz():
    return jsonify({"status": "ok"})


@app.post("/api/predict")
def predict():
    images = [image for image in request.files.getlist("image") if image and image.filename]
    model_name = request.form.get("model")
    attention_type = request.form.get("attention")

    if not images:
        return jsonify({"error": "Please upload at least one image first."}), 400
    if not model_name:
        return jsonify({"error": "Please choose a model."}), 400
    if not attention_type:
        return jsonify({"error": "Please choose an attention option."}), 400

    try:
        results = []
        for image in images:
            image_path = save_upload_to_temp(image)
            result = predict_image(image_path, model_name, attention_type)
            results.append(
                {
                    "filename": image.filename,
                    "predictedClass": result.predicted_class,
                    "confidence": result.confidence,
                    "probabilities": result.probabilities,
                    "previewImage": result.preview_image,
                    "cams": {
                        "gradcam": result.gradcam,
                        "gradcampp": result.gradcampp,
                        "scorecam": result.scorecam,
                    },
                }
            )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        return jsonify({"error": str(exc)}), 500

    return jsonify({"count": len(results), "results": results})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    debug = os.environ.get("FLASK_DEBUG", "1") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
