import wave
from pathlib import Path

import joblib
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "saved_models" / "audio_deepfake_model.pkl"
AUDIO_DIR = ROOT_DIR / "data" / "raw" / "audios"

def read_wav_samples(file_path):
    try:
        from scipy.io import wavfile

        sample_rate, samples = wavfile.read(str(file_path))
        samples = np.asarray(samples)
        if samples.ndim > 1:
            samples = samples.mean(axis=1)
        if np.issubdtype(samples.dtype, np.integer):
            max_value = np.iinfo(samples.dtype).max or 1
            samples = samples.astype(np.float32) / max_value
        else:
            samples = samples.astype(np.float32)
    except Exception:
        with wave.open(str(file_path), "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            frame_count = wav_file.getnframes()
            raw = wav_file.readframes(frame_count)

        if sample_width != 2:
            raise ValueError("Only PCM/float WAV audio is supported by the fast audio model.")

        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
        if channels > 1:
            samples = samples.reshape(-1, channels).mean(axis=1)
        samples = samples / 32768.0

    if samples.size == 0:
        raise ValueError("Audio file has no samples.")

    samples = np.nan_to_num(samples)
    return samples, sample_rate


def extract_fast_audio_details(file_path):
    samples, sample_rate = read_wav_samples(file_path)
    window = samples[: min(samples.size, sample_rate * 10)]
    spectrum = np.abs(np.fft.rfft(window))
    bands = np.array_split(spectrum, 13)
    mfcc_like = np.array([np.log1p(np.mean(band)) for band in bands])
    chroma_like = np.array([
        np.mean(spectrum[index::12]) if spectrum[index::12].size else 0
        for index in range(12)
    ])
    chroma_like = chroma_like / (np.max(chroma_like) or 1)
    spectral_bands = np.array_split(spectrum, 7)
    spectral_contrast_like = np.array([
        np.log1p(np.percentile(band, 95) - np.percentile(band, 5))
        for band in spectral_bands
    ])
    combined_features = np.hstack([mfcc_like, chroma_like, spectral_contrast_like])

    return {
        "sample_rate": int(sample_rate),
        "duration_seconds": round(float(samples.size / sample_rate), 3) if sample_rate else 0,
        "feature_count": int(combined_features.size),
        "mfcc": [round(float(value), 6) for value in mfcc_like],
        "chroma": [round(float(value), 6) for value in chroma_like],
        "spectral_contrast": [round(float(value), 6) for value in spectral_contrast_like],
        "combined": [round(float(value), 6) for value in combined_features],
    }


def fallback_byte_details(file_path):
    raw = Path(file_path).read_bytes()
    values = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
    if values.size == 0:
        values = np.zeros(1, dtype=np.float32)
    normalized = (values - 128) / 128
    spectrum = np.abs(np.fft.rfft(normalized[: min(normalized.size, 65536)]))
    mfcc_like = np.array([np.mean(chunk) for chunk in np.array_split(normalized, 13)])
    chroma_like = np.array([np.mean(np.abs(normalized[index::12])) for index in range(12)])
    chroma_like = chroma_like / (np.max(chroma_like) or 1)
    spectral_contrast_like = np.array([np.log1p(np.mean(band)) for band in np.array_split(spectrum, 7)])
    combined_features = np.hstack([mfcc_like, chroma_like, spectral_contrast_like])

    return {
        "sample_rate": 0,
        "duration_seconds": 0,
        "feature_count": int(combined_features.size),
        "decoder_warning": "Fast WAV decoding unavailable; used byte-level fallback features.",
        "mfcc": [round(float(value), 6) for value in mfcc_like],
        "chroma": [round(float(value), 6) for value in chroma_like],
        "spectral_contrast": [round(float(value), 6) for value in spectral_contrast_like],
        "combined": [round(float(value), 6) for value in combined_features],
    }


def feature_vector(file_path):
    try:
        details = extract_fast_audio_details(file_path)
    except Exception:
        details = fallback_byte_details(file_path)

    mfcc = np.asarray(details["mfcc"], dtype=float)
    chroma = np.asarray(details["chroma"], dtype=float)
    contrast = np.asarray(details["spectral_contrast"], dtype=float)
    combined = np.asarray(details["combined"], dtype=float)

    summary = np.asarray([
        details["sample_rate"] / 48000,
        details["duration_seconds"] / 30,
        np.mean(mfcc),
        np.std(mfcc),
        np.mean(chroma),
        np.std(chroma),
        np.mean(contrast),
        np.std(contrast),
        np.mean(combined),
        np.std(combined),
    ], dtype=float)

    return np.hstack([combined, summary]), details


def labeled_audio_files(audio_dir=AUDIO_DIR):
    files = []
    if not audio_dir.exists():
        return files

    for path in sorted(audio_dir.glob("*.wav")):
        lower_name = path.name.lower()
        if lower_name.startswith("fake"):
            files.append((path, 1))
        elif lower_name.startswith("real"):
            files.append((path, 0))

    return files


def train_audio_model():
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rows = labeled_audio_files()
    if len(rows) < 4:
        raise RuntimeError("Need at least four labeled audio files in data/raw/audios.")

    vectors = []
    labels = []
    for path, label in rows:
        vector, _ = feature_vector(path)
        vectors.append(vector)
        labels.append(label)

    X = np.vstack(vectors)
    y = np.asarray(labels)
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
    )

    folds = min(5, np.bincount(y).min())
    cv_accuracy = None
    if folds >= 2:
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
        cv_accuracy = float(np.mean(cross_val_score(model, X, y, cv=splitter)))

    model.fit(X, y)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": model,
        "feature_count": int(X.shape[1]),
        "training_samples": int(X.shape[0]),
        "cv_accuracy": cv_accuracy,
    }, MODEL_PATH)

    return MODEL_PATH


def load_audio_model():
    if not MODEL_PATH.exists():
        train_audio_model()
    return joblib.load(MODEL_PATH)


def predict_audio(file_path):
    vector, details = feature_vector(file_path)

    try:
        bundle = load_audio_model()
        model = bundle["model"]

        if hasattr(model, "predict_proba"):
            fake_probability = float(model.predict_proba([vector])[0][1])
        else:
            fake_probability = float(model.predict([vector])[0])

        return {
            "fake_probability": max(0.02, min(0.98, fake_probability)),
            "features": details,
            "training_samples": bundle.get("training_samples"),
            "cv_accuracy": bundle.get("cv_accuracy"),
            "feature_count": bundle.get("feature_count"),
        }
    except Exception as exc:
        combined = np.asarray(details["combined"], dtype=float)
        mfcc = np.asarray(details["mfcc"], dtype=float)
        chroma = np.asarray(details["chroma"], dtype=float)
        contrast = np.asarray(details["spectral_contrast"], dtype=float)

        energy = float(np.mean(np.abs(combined)))
        tonal_spread = float(np.std(chroma))
        contrast_spread = float(np.std(contrast))
        cepstral_spread = float(np.std(mfcc))
        duration_score = min(details.get("duration_seconds", 0) / 30, 1)

        fake_probability = 0.32
        fake_probability += min(energy / 0.45, 1) * 0.16
        fake_probability += min(tonal_spread / 0.32, 1) * 0.18
        fake_probability += min(contrast_spread / 1.2, 1) * 0.16
        fake_probability += min(cepstral_spread / 0.8, 1) * 0.12
        fake_probability += duration_score * 0.06

        details["model_warning"] = f"Trained audio model unavailable; used acoustic fallback: {exc}"

        return {
            "fake_probability": max(0.08, min(0.92, fake_probability)),
            "features": details,
            "training_samples": 0,
            "cv_accuracy": None,
            "feature_count": int(vector.size),
        }
