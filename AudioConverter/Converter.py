import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
from scipy.signal import stft

DIR = 'D:/Documents/GoogleDrive/STUDY/SEM5/PK/AudioCapture/'
fns = [
    # 'Jacek/correct/31-10-19_08-18-13.wav',
    'Sebastian/correct/30-10-19_11-59-51.wav',
    # 'Alicja/correct/18-10-19_15-13-50.wav',
    # 'Alicja/incorrect/18-10-19_15-25-09.wav',
    # 'Jacek/incorrect/31-10-19_07-43-34.wav'
    'Sebastian/incorrect/30-10-19_12-00-07.wav'
]

# 'Alicja/incorrect/18-10-19_14-59-48.wav'
# 'Alicja/correct/18-10-19_15-13-50.wav'
SAMPLE_RATE = 16000


def read_wav_file(x):
    # Read wavfile using scipy wavfile.read
    _, wav = wavfile.read(x)
    # Normalize
    wav = wav.astype(np.float32) / np.iinfo(np.int8).max

    return wav


def log_spectrogram(wav):
    freqs, times, spec = stft(wav, SAMPLE_RATE, nperseg=400, noverlap=350, nfft=512,
                              padded=False, boundary=None)
    # Log spectrogram
    amp = np.log(np.abs(spec) + 1e-10)

    return freqs, times, amp


fig = [plt.figure(figsize=(14, 8)), plt.figure(figsize=(14, 8))]
for i, fn in enumerate(fns):
    wav = read_wav_file(DIR + fn)
    if len(wav.shape) == 2:
        wav = wav.sum(axis=1) / 2

    freqs, times, amp = log_spectrogram(wav)
    ax = fig[0].add_subplot(len(fns), 1, i + 1)
    ax.set_title('Surowy wave dla ' + fn)
    ax.set_ylabel('Amplituda')
    ax.set_xlabel('Czas w s')
    ax.plot(np.linspace(0, len(wav) / SAMPLE_RATE, len(wav)), wav)

    ax2 = fig[1].add_subplot(len(fns), 1, i + 1)
    ax2.imshow(amp, aspect='auto', origin='lower',
               extent=[times.min(), times.max(), freqs.min(), freqs.max()])
    ax2.set_title('Spektrogram dla ' + fn)
    ax2.set_ylabel('Częstotliwość w Hz')
    ax2.set_xlabel('Czas w s')
    plt.xlim(left=0, right=3)
for f in fig:
    f.tight_layout()
    f.show()
