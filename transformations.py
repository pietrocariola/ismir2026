import librosa

def identity(y, sr):
    return y, sr

def make_runner(n_steps):
    def pitch_shift(y, sr):
        # 1 step is a semi-tone
        y = librosa.effects.pitch_shift(
            y,
            sr=sr,
            n_steps=1
        )
        return y, sr
    return pitch_shift

pitch_shift_1 = make_runner(1)
pitch_shift_2 = make_runner(2)
pitch_shift_3 = make_runner(3)
pitch_shift_4 = make_runner(4)
pitch_shift_5 = make_runner(5)
pitch_shift_6 = make_runner(6)
pitch_shift_7 = make_runner(7)
pitch_shift_8 = make_runner(8)
pitch_shift_9 = make_runner(9)
pitch_shift_10 = make_runner(10)
pitch_shift_11 = make_runner(11)
pitch_shift_12 = make_runner(12)  

def make_runner(rate):
    def time_stretch(y, sr):
        # rate 1.20 is 20% faster
        # rate 0.80 is 20% slower
        y = librosa.effects.time_stretch(y, rate=rate)
        return y, sr
    return time_stretch

time_stretch_11 = make_runner(1.1)
time_stretch_12 = make_runner(1.2)
time_stretch_13 = make_runner(1.3)
time_stretch_14 = make_runner(1.4)
time_stretch_15 = make_runner(1.5)
time_stretch_20 = make_runner(2.0)

time_stretch_095 = make_runner(0.95)
time_stretch_090 = make_runner(0.90)
time_stretch_085 = make_runner(0.85)
time_stretch_080 = make_runner(0.80)
time_stretch_075 = make_runner(0.75)
time_stretch_050 = make_runner(0.50)

tf_dict = {
    "identity": identity,
}