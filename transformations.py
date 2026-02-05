import librosa
import numpy as np

# no transformation
def identity(y: np.ndarray, sr: int, *args):
    return y, sr

def pitchshift(y: np.ndarray, sr: int, n_steps: int):
    # 1 step is a semi-tone
    y = librosa.effects.pitch_shift(
        y,
        sr=sr,
        n_steps=n_steps
    )
    return y, sr

def timestretch(y: np.ndarray, sr: int, rate: float):
    # rate 1.20 is 20% faster
    # rate 0.80 is 20% slower
    y = librosa.effects.time_stretch(y, rate=rate)
    return y, sr

tf_dict = {
    "identity": identity,
    "pitchshift": pitchshift,
    "timestretch": timestretch,
}

tf_dict_params = {
    "identity": ["none"],
    "pitchshift": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 24,
                    -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -24],
    "timestretch": [1.1, 1.2, 1.3, 1.4, 1.5, 2.0,
                     1/1.1, 1/1.2, 1/1.3, 1/1.4, 1/1.5, 1/2.0],
}