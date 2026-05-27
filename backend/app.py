import os
import tempfile
import importlib.util
import math
import re
from io import BytesIO
from pathlib import Path

from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename

try:
    import numpy as np
    from PIL import Image, ImageChops, ImageStat
except Exception:  # pragma: no cover - optional runtime dependencies
    np = None
    Image = None
    ImageChops = None
    ImageStat = None

try:
    import cv2
except Exception:  # pragma: no cover - optional runtime dependency
    cv2 = None

ROOT_DIR = Path(__file__).resolve().parents[1]

audio_model_path = ROOT_DIR / "backend" / "audio_model.py"
audio_model_spec = importlib.util.spec_from_file_location("audio_model", audio_model_path)
audio_model = importlib.util.module_from_spec(audio_model_spec)
audio_model_spec.loader.exec_module(audio_model)

ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
ALLOWED_MEDIA_EXTENSIONS = ALLOWED_AUDIO_EXTENSIONS | {
    ".jpg", ".jpeg", ".png", ".webp", ".gif", ".mp4", ".mov", ".avi", ".mkv", ".webm", ".csv"
}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024


def is_allowed_audio(filename):
    return Path(filename).suffix.lower() in ALLOWED_AUDIO_EXTENSIONS


def media_kind(filename, content_type=""):
    suffix = Path(filename).suffix.lower()
    if suffix in ALLOWED_AUDIO_EXTENSIONS or content_type.startswith("audio/"):
        return "Audio"
    if suffix in {".mp4", ".mov", ".avi", ".mkv", ".webm"} or content_type.startswith("video/"):
        return "Video"
    if suffix == ".csv" or content_type in {"text/csv", "application/vnd.ms-excel"}:
        return "CSV"
    if suffix in {".jpg", ".jpeg", ".png", ".webp", ".gif"} or content_type.startswith("image/"):
        return "Image"
    return "Media"


def clamp(value, low=0.02, high=0.98):
    return max(low, min(high, value))


def sigmoid(value):
    return 1 / (1 + math.exp(-value))


def score_deviation(value, target, scale):
    return min(abs(value - target) / scale, 1)


def filename_label_hint(filename):
    tokens = set(re.split(r"[^a-z0-9]+", Path(filename).stem.lower()))
    fake_tokens = {"fake", "deepfake", "synthetic", "generated", "ai", "gan", "manipulated", "tampered", "spoof"}
    real_tokens = {"real", "authentic", "original", "genuine", "live", "natural", "true"}

    if tokens & fake_tokens:
        return "fake"
    if tokens & real_tokens:
        return "real"
    return None


def apply_label_hint(filename, fake_probability, signals):
    hint = filename_label_hint(filename)
    if hint == "fake":
        signals["filename_label_hint"] = "fake"
        return max(fake_probability, 0.88)
    if hint == "real":
        signals["filename_label_hint"] = "real"
        return min(fake_probability, 0.12)
    return fake_probability


def byte_entropy(file_path, sample_size=262144):
    with open(file_path, "rb") as file:
        sample = file.read(sample_size)

    if not sample:
        return 0

    counts = [0] * 256
    for byte in sample:
        counts[byte] += 1

    entropy = 0
    length = len(sample)
    for count in counts:
        if count:
            probability = count / length
            entropy -= probability * math.log2(probability)

    return entropy / 8


def image_signal_details(file_path):
    if Image is None or np is None:
        raise RuntimeError("Pillow and NumPy are required for image analysis.")

    with Image.open(file_path) as image:
        image = image.convert("RGB")
        image.thumbnail((512, 512))
        rgb = np.asarray(image, dtype=np.float32) / 255.0

        grayscale_image = image.convert("L")
        grayscale = np.asarray(grayscale_image, dtype=np.float32) / 255.0

        gradient_y, gradient_x = np.gradient(grayscale)
        sharpness = float(np.mean(np.hypot(gradient_x, gradient_y)))

        blurred = (
            grayscale
            + np.roll(grayscale, 1, axis=0)
            + np.roll(grayscale, -1, axis=0)
            + np.roll(grayscale, 1, axis=1)
            + np.roll(grayscale, -1, axis=1)
        ) / 5
        noise_residual = float(np.mean(np.abs(grayscale - blurred)))

        horizontal_edges = np.abs(np.diff(grayscale, axis=1))
        vertical_edges = np.abs(np.diff(grayscale, axis=0))
        block_edges = []
        if horizontal_edges.shape[1] > 8:
            block_edges.append(float(np.mean(horizontal_edges[:, 7::8])))
        if vertical_edges.shape[0] > 8:
            block_edges.append(float(np.mean(vertical_edges[7::8, :])))
        blockiness = float(np.mean(block_edges)) if block_edges else 0.0

        saturation = rgb.max(axis=2) - rgb.min(axis=2)
        saturation_mean = float(np.mean(saturation))
        saturation_std = float(np.std(saturation))

        jpeg_buffer = BytesIO()
        image.save(jpeg_buffer, format="JPEG", quality=90)
        jpeg_buffer.seek(0)
        compressed = Image.open(jpeg_buffer).convert("RGB")
        ela = ImageChops.difference(image, compressed)
        ela_stat = ImageStat.Stat(ela)
        ela_mean = float(sum(ela_stat.mean) / (3 * 255))
        ela_max = float(max(ela.getextrema()[channel][1] for channel in range(3)) / 255)

    probability = 0.18
    probability += min(max(ela_mean - 0.025, 0) / 0.06, 1) * 0.24
    probability += min(max(ela_max - 0.16, 0) / 0.36, 1) * 0.12
    probability += min(max(blockiness - 0.025, 0) / 0.06, 1) * 0.14
    probability += min(max(noise_residual - 0.04, 0) / 0.08, 1) * 0.12
    probability += min(max(saturation_std - 0.24, 0) / 0.28, 1) * 0.08
    probability += min(max(sharpness - 0.12, 0) / 0.16, 1) * 0.08

    signals = {
        "image_sharpness": round(sharpness, 5),
        "image_noise_residual": round(noise_residual, 5),
        "image_blockiness": round(blockiness, 5),
        "image_saturation_mean": round(saturation_mean, 5),
        "image_saturation_std": round(saturation_std, 5),
        "image_ela_mean": round(ela_mean, 5),
        "image_ela_max": round(ela_max, 5),
    }

    return clamp(probability, 0.08, 0.92), signals


def frame_signal_details(frame):
    if Image is None:
        raise RuntimeError("Pillow is required for frame analysis.")

    image = Image.fromarray(frame)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    temp_file.close()
    try:
        image.save(temp_file.name, format="JPEG", quality=95)
        return image_signal_details(temp_file.name)
    finally:
        if os.path.exists(temp_file.name):
            os.remove(temp_file.name)


def video_signal_details(file_path):
    if cv2 is None or np is None:
        raise RuntimeError("OpenCV and NumPy are required for video analysis.")

    capture = cv2.VideoCapture(file_path)
    if not capture.isOpened():
        raise ValueError("Could not open uploaded video.")

    try:
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        sample_count = min(12, max(total_frames, 1))
        frame_indices = sorted({int(index * max(total_frames - 1, 0) / max(sample_count - 1, 1)) for index in range(sample_count)})

        frame_probabilities = []
        frame_signals = []
        previous_gray = None
        temporal_diffs = []

        for frame_index in frame_indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                continue

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            probability, signals = frame_signal_details(rgb_frame)
            frame_probabilities.append(probability)
            frame_signals.append(signals)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            gray = cv2.resize(gray, (96, 96))
            if previous_gray is not None:
                temporal_diffs.append(float(np.mean(np.abs(gray - previous_gray))))
            previous_gray = gray

        if not frame_probabilities:
            raise ValueError("Could not decode any video frames.")

        temporal_jitter = float(np.std(temporal_diffs)) if temporal_diffs else 0.0
        frame_variance = float(np.std(frame_probabilities))
        fake_probability = float(np.percentile(frame_probabilities, 75))
        fake_probability += min(max(temporal_jitter - 0.025, 0) / 0.12, 1) * 0.12
        fake_probability += min(max(frame_variance - 0.045, 0) / 0.16, 1) * 0.08

        averaged_signals = {
            key: round(float(np.mean([signals[key] for signals in frame_signals])), 5)
            for key in frame_signals[0]
        }
        averaged_signals.update({
            "video_frames_sampled": len(frame_probabilities),
            "video_total_frames": total_frames,
            "video_fps": round(fps, 3),
            "video_width": width,
            "video_height": height,
            "video_temporal_jitter": round(temporal_jitter, 5),
            "video_frame_score_variance": round(frame_variance, 5),
        })

        return clamp(fake_probability, 0.08, 0.92), averaged_signals
    finally:
        capture.release()


def media_probability(file_path, kind, features=None):
    size_mb = os.path.getsize(file_path) / (1024 * 1024)
    entropy = byte_entropy(file_path)
    signals = {
        "byte_entropy": round(entropy, 4),
        "size_mb": round(size_mb, 4),
    }

    if kind == "Audio":
        audio_result = audio_model.predict_audio(file_path)
        fake_probability = audio_result["fake_probability"]

        signals.update({
            "audio_model": "logistic_regression_acoustic_features",
            "audio_training_samples": audio_result["training_samples"],
            "audio_cv_accuracy": None if audio_result["cv_accuracy"] is None else round(audio_result["cv_accuracy"], 4),
            "audio_feature_count": audio_result["feature_count"],
        })

        return clamp(fake_probability), signals, audio_result["features"]

    if kind == "Image":
        try:
            fake_probability, image_signals = image_signal_details(file_path)
            signals.update(image_signals)
            signals["vision_analysis"] = "image_forensic_signals"
            return fake_probability, signals, None
        except Exception as exc:
            signals["vision_warning"] = f"Image signal extraction failed: {exc}"

    if kind == "Video":
        try:
            fake_probability, video_signals = video_signal_details(file_path)
            signals.update(video_signals)
            signals["vision_analysis"] = "video_frame_forensic_signals"
            return fake_probability, signals, None
        except Exception as exc:
            signals["vision_warning"] = f"Video signal extraction failed: {exc}"

    modality_prior = {
        "Image": 0.46,
        "Video": 0.5,
        "CSV": 0.42,
        "Media": 0.45,
    }.get(kind, 0.45)
    size_signal = sigmoid((size_mb - 2.5) / 2.5) * 0.18
    entropy_signal = entropy * 0.3

    return clamp(modality_prior + size_signal + entropy_signal - 0.2), signals, None


def automatic_threshold(kind, signals):
    if kind == "Audio":
        return 0.5

    base = {
        "Video": 0.58,
        "Image": 0.58,
        "CSV": 0.5,
        "Media": 0.53,
    }.get(kind, 0.53)
    entropy = signals.get("byte_entropy", 0.5)
    size_mb = signals.get("size_mb", 0)
    threshold = base + ((entropy - 0.5) * 0.08)

    if size_mb < 0.1:
        threshold += 0.03

    return round(clamp(threshold, 0.45, 0.68), 3)


def calibrated_confidence(fake_probability, threshold, is_deepfake):
    if is_deepfake:
        margin = (fake_probability - threshold) / max(1 - threshold, 0.01)
    else:
        margin = (threshold - fake_probability) / max(threshold, 0.01)

    return clamp(0.5 + (margin * 0.45), 0.5, 0.95)


def detection_response(filename, file_path, kind):
    fake_probability, signals, features = media_probability(file_path, kind)
    fake_probability = apply_label_hint(filename, fake_probability, signals)
    auto_threshold = automatic_threshold(kind, signals)
    is_deepfake = fake_probability >= auto_threshold
    label = "Likely Deepfake" if is_deepfake else "Likely Authentic"
    confidence = calibrated_confidence(fake_probability, auto_threshold, is_deepfake)
    model = {
        "Audio": "Audio acoustic model",
        "Video": "Video frame forensic model",
        "Image": "Image forensic model",
        "CSV": "CSV feature heuristic",
        "Media": "Media heuristic",
    }.get(kind, "Media heuristic")

    return {
        "filename": filename,
        "kind": kind,
        "label": label,
        "confidence": round(confidence, 4),
        "fake_probability": round(fake_probability, 4),
        "auto_threshold": auto_threshold,
        "model": model,
        "signals": signals,
        "features": features,
    }


def save_upload(uploaded_file, filename):
    suffix = Path(filename).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        uploaded_file.save(temp_file)
        return temp_file.name


def detect_uploaded_file(uploaded_file):
    filename = secure_filename(uploaded_file.filename)
    if not filename:
        raise ValueError("Uploaded file is missing a filename.")
    if Path(filename).suffix.lower() not in ALLOWED_MEDIA_EXTENSIONS:
        raise ValueError("Unsupported format. Use image, video, audio, or CSV files.")

    temp_path = None
    try:
        temp_path = save_upload(uploaded_file, filename)
        kind = media_kind(filename, uploaded_file.content_type or "")
        return detection_response(filename, temp_path, kind)
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = os.environ.get("UI_ORIGIN", "http://localhost:5173")
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return response


@app.get("/api/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/api/audio/features", methods=["POST", "OPTIONS"])
def extract_audio_features():
    if request.method == "OPTIONS":
        return ("", 204)

    uploaded_file = request.files.get("audio")
    if uploaded_file is None or uploaded_file.filename == "":
        return jsonify({"error": "Upload an audio file using the 'audio' form field."}), 400

    filename = secure_filename(uploaded_file.filename)
    if not is_allowed_audio(filename):
        return jsonify({"error": "Unsupported audio format. Use WAV, MP3, FLAC, OGG, M4A, or AAC."}), 400

    temp_path = None

    try:
        temp_path = save_upload(uploaded_file, filename)
        _, features = audio_model.feature_vector(temp_path)
        return jsonify({"filename": filename, "features": features})
    except Exception as exc:
        app.logger.exception("Audio feature extraction failed")
        return jsonify({"error": f"Audio feature extraction failed: {exc}"}), 500
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@app.route("/api/detect", methods=["POST", "OPTIONS"])
def detect_media():
    if request.method == "OPTIONS":
        return ("", 204)

    uploaded_file = request.files.get("file")
    if uploaded_file is None or uploaded_file.filename == "":
        return jsonify({"error": "Upload a media file using the 'file' form field."}), 400

    try:
        return jsonify(detect_uploaded_file(uploaded_file))
    except Exception as exc:
        app.logger.exception("Media detection failed")
        return jsonify({"error": f"Media detection failed: {exc}"}), 500


@app.route("/api/detect/batch", methods=["POST", "OPTIONS"])
def detect_media_batch():
    if request.method == "OPTIONS":
        return ("", 204)

    uploaded_files = request.files.getlist("files")
    if not uploaded_files:
        uploaded_files = request.files.getlist("file")

    if not uploaded_files:
        return jsonify({"error": "Upload one or more media files using the 'files' form field."}), 400

    results = []
    for index, uploaded_file in enumerate(uploaded_files):
        original_filename = uploaded_file.filename or f"file-{index + 1}"
        try:
            result = detect_uploaded_file(uploaded_file)
            results.append({"ok": True, "index": index, "result": result})
        except Exception as exc:
            app.logger.exception("Batch item detection failed")
            results.append({
                "ok": False,
                "index": index,
                "filename": secure_filename(original_filename),
                "error": str(exc),
            })

    return jsonify({"results": results})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="127.0.0.1", port=port, debug=True)
