import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
from scipy.signal import stft

DIR = 'D:/Documents/AudioCapture'
fns = ['/correct/30-10-19_08-22-30.wav',
       '/correct/30-10-19_08-22-33.wav',
       '/incorrect/30-10-19_08-22-49.wav']
SAMPLE_RATE = 16000


def read_wav_file(x):
    # Read wavfile using scipy wavfile.read
    _, wav = wavfile.read(x)
    # Normalize
    wav = wav.astype(np.float32) / np.iinfo(np.int16).max

    return wav


fig = plt.figure(figsize=(14, 8))
for i, fn in enumerate(fns):
    wav = read_wav_file(DIR + fn)

    ax = fig.add_subplot(3, 1, i + 1)
    ax.set_title('Raw wave of ' + fn)
    ax.set_ylabel('Amplitude')
    plt.ylim(top=0.5, bottom=-0.5)
    ax.plot(np.linspace(0, len(wav)/SAMPLE_RATE, len(wav)), wav)
fig.tight_layout()
fig.show()


def log_spectrogram(wav):
    freqs, times, spec = stft(wav, SAMPLE_RATE, nperseg=400, noverlap=200, nfft=512,
                              padded=True, boundary='zeros')
    # Log spectrogram
    amp = np.log(np.abs(spec) + 1e-10)

    return freqs, times, amp


fig = plt.figure(figsize=(14, 8))
for i, fn in enumerate(fns):
    wav = read_wav_file(DIR + fn)
    freqs, times, amp = log_spectrogram(wav)

    ax = fig.add_subplot(3, 1, i + 1)
    ax.imshow(amp, aspect='auto', origin='lower',
              extent=[times.min(), times.max(), freqs.min(), freqs.max()])
    ax.set_title('Spectrogram of ' + fn)
    ax.set_ylabel('Freqs in Hz')
    ax.set_xlabel('Seconds')
fig.tight_layout()
fig.show()