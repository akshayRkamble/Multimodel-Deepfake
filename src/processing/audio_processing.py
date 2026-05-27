import librosa
import numpy as np
import logging
import wave

def load_audio(file_path):
    """
    Loading audio file.
    :param file_path: Path to the audio file
    :return: Audio time series and sampling rate
    """
    try:
        y, sr = librosa.load(file_path, sr=None)
        logging.info(f"Audio file loaded: {file_path}")
        return y, sr
    except Exception as e:
        logging.error(f"Error loading audio file {file_path}: {e}")
        raise

def extract_mfcc(y, sr, n_mfcc=13):
    """
    Extracting MFCC features from audio time series.
    :param y: Audio time series
    :param sr: Sampling rate of the audio
    :param n_mfcc: Number of MFCCs to return
    :return: Mean MFCC features
    """
    try:
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfcc_mean = np.mean(mfcc.T, axis=0)
        logging.info("MFCC features extracted")
        return mfcc_mean
    except Exception as e:
        logging.error(f"Error extracting MFCC features: {e}")
        raise

def extract_chroma(y, sr):
    """
    Extracting chroma features from audio time series.
    :param y: Audio time series
    :param sr: Sampling rate of the audio
    :return: Mean chroma features
    """
    try:
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma.T, axis=0)
        logging.info("Chroma features extracted")
        return chroma_mean
    except Exception as e:
        logging.error(f"Error extracting chroma features: {e}")
        raise

def extract_spectral_contrast(y, sr):
    """
    Extracting spectral contrast features from audio time series.
    :param y: Audio time series
    :param sr: Sampling rate of the audio
    :return: Mean spectral contrast features
    """
    try:
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        spectral_contrast_mean = np.mean(spectral_contrast.T, axis=0)
        logging.info("Spectral contrast features extracted")
        return spectral_contrast_mean
    except Exception as e:
        logging.error(f"Error extracting spectral contrast features: {e}")
        raise

def process_audio(file_path):
    """
    Processing an audio file and extracting features.
    :param file_path: Path to the audio file
    :return: Extracted audio features
    """
    try:
        y, sr = load_audio(file_path)
        mfcc_features = extract_mfcc(y, sr)
        chroma_features = extract_chroma(y, sr)
        spectral_contrast_features = extract_spectral_contrast(y, sr)
        
        audio_features = np.hstack([mfcc_features, chroma_features, spectral_contrast_features])
        logging.info(f"Extracted features from audio file: {file_path}")
        
        return audio_features
    except Exception as e:
        logging.error(f"Error processing audio file {file_path}: {e}")
        raise

def process_audio_details(file_path):
    """
    Processing an audio file and returning JSON-friendly feature groups.
    :param file_path: Path to the audio file
    :return: Audio metadata and extracted audio features
    """
    try:
        y, sr = load_audio(file_path)
        mfcc_features = extract_mfcc(y, sr)
        chroma_features = extract_chroma(y, sr)
        spectral_contrast_features = extract_spectral_contrast(y, sr)
        combined_features = np.hstack([mfcc_features, chroma_features, spectral_contrast_features])

        return {
            "sample_rate": int(sr),
            "duration_seconds": round(float(len(y) / sr), 3) if sr else 0,
            "feature_count": int(combined_features.size),
            "mfcc": [round(float(value), 6) for value in mfcc_features],
            "chroma": [round(float(value), 6) for value in chroma_features],
            "spectral_contrast": [round(float(value), 6) for value in spectral_contrast_features],
            "combined": [round(float(value), 6) for value in combined_features],
        }
    except Exception as e:
        logging.error(f"Error processing audio details for {file_path}: {e}")
        raise

def process_audio_details_fast(file_path):
    """
    Fast JSON-friendly audio features for WAV files using the Python standard library.
    Falls back to the librosa extractor for compressed formats.
    """
    if not file_path.lower().endswith(".wav"):
        return process_audio_details(file_path)

    try:
        with wave.open(file_path, "rb") as wav_file:
            sr = wav_file.getframerate()
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            frame_count = wav_file.getnframes()
            raw = wav_file.readframes(frame_count)

        if sample_width != 2:
            return process_audio_details(file_path)

        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
        if channels > 1:
            samples = samples.reshape(-1, channels).mean(axis=1)
        samples = samples / 32768.0

        if samples.size == 0:
            raise ValueError("Audio file has no samples.")

        window = samples[: min(samples.size, sr * 10)]
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
            "sample_rate": int(sr),
            "duration_seconds": round(float(samples.size / sr), 3) if sr else 0,
            "feature_count": int(combined_features.size),
            "mfcc": [round(float(value), 6) for value in mfcc_like],
            "chroma": [round(float(value), 6) for value in chroma_like],
            "spectral_contrast": [round(float(value), 6) for value in spectral_contrast_like],
            "combined": [round(float(value), 6) for value in combined_features],
        }
    except Exception as e:
        logging.warning(f"Fast WAV processing failed for {file_path}; falling back to librosa: {e}")
        return process_audio_details(file_path)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python audio_processing.py <path_to_audio_file>")
        sys.exit(1)

    file_path = sys.argv[1]
    features = process_audio(file_path)
    print("Extracted Features:\n", features)
