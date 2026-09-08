import librosa
import numpy as np
import essentia
import essentia.standard as es
from audiomentations import BitCrush

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

def highpass(y: np.ndarray, sr: int, cutoff: int):
    y = es.HighPass(cutoffFrequency=cutoff, sampleRate=sr)(y)
    return y, sr

def lowpass(y: np.ndarray, sr: int, cutoff: int):
    y = es.LowPass(cutoffFrequency=cutoff, sampleRate=sr)(y)
    return y, sr

def clipper(y: np.ndarray, sr: int, clip_level: float):
    y = es.Clipper(max=clip_level, min=-clip_level)(y)
    return y, sr

def noiseadder(y: np.ndarray, sr: int, level: int):
    # level = -100dB low noise
    # level = 0dB a lot of noise
    # fixSeed=True to fix seed as 0
    y = es.NoiseAdder(fixSeed=True, level=level)(y)
    return y, sr

def bitcrush(y: np.ndarray, sr: int, depth: int):
    y = BitCrush(min_bit_depth=depth, max_bit_depth=depth, p=1.0)(y, sr)
    return y, sr

def gain(y: np.ndarray, sr: int, gain: float):
    y = y*gain
    return y, sr

tf_dict = {
    "identity": identity,
    "pitchshift": pitchshift,
    "timestretch": timestretch,
    "highpass": highpass,
    "lowpass": lowpass,
    "clipper": clipper,
    "noiseadder": noiseadder,
    "bitcrush": bitcrush,
    "gain": gain,
}

tf_dict_params = {
    "identity": ["none"],
    "pitchshift": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 24,
                    -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -24],
    "timestretch": [1.1, 1.2, 1.3, 1.4, 1.5, 2.0,
                     1/1.1, 1/1.2, 1/1.3, 1/1.4, 1/1.5, 1/2.0],
    "highpass": [500, 1000, 2000, 4000, 8000, 16000],
    "lowpass": [16000, 8000, 4000, 2000, 1000, 500],
    "clipper": [0.1, 0.2, 0.4, 0.8],
    "noiseadder": [-100, -80, -60, -40, -20, 0],
    "bitcrush": [4, 6, 8, 10, 12, 14],
    "gain": [1.2, 1.5, 2.0, 3.0, 1/1.2, 1/1.5, 1/2.0, 1/3.0],
}

tf_dict_ids = {
    "pitchshift": 0,
    "timestretch": 1,
    "highpass": 0,
    "lowpass": 20000,
    "clipper": 1,
    "noiseadder": -200,
    "bitcrush": 16,
    "gain": 1,
}

tf_dict_ticks = {
    "pitchshift": [-24, -12, 0, 12, 24],
    "timestretch": [0, 0.5, 1, 1.5, 2],
    "highpass": [0, 4000, 8000, 12000, 16000],
    "lowpass": [0, 4000, 8000, 12000, 16000],
    "clipper": [0, 0.25, 0.5, 0.75, 1],
    "noiseadder": [-200, -150, -100, -50, 0],
    "bitcrush": [4, 6, 8, 10, 12, 14, 16],
    "gain": [0, 1, 2, 3],
}

tf_dict_scale = {
    "pitchshift": "semitones",
    "timestretch": "fraction",
    "highpass": "cutoff (Hz)",
    "lowpass": "cutoff (Hz)",
    "clipper": "ceiling",
    "noiseadder": "noise dB",
    "bitcrush": "bit depth",
    "gain": "fraction",
}