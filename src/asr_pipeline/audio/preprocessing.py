import numpy as np
import librosa

def resample_audio(audio: np.ndarray, sample_rate: int, target_rate:int) -> np.ndarray:
        if sample_rate != target_rate:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=target_rate)
        return audio