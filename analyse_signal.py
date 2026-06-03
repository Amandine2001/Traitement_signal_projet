import librosa
import numpy as np
import matplotlib.pyplot as plt


def calculer_et_visualiser_fft(
    filename: str, name_plot: str = "Spectre de fréquences", max_freq: int = 5000
):
    y, sr = librosa.load(filename, sr=None)
    N = len(y)
    Y = np.fft.fft(y)
    frequencies = np.fft.fftfreq(N, d=1 / sr)
    amplitude = np.abs(Y)
    # Affichage du spectre de fréquences
    plt.figure(figsize=(14, 5))
    plt.plot(
        frequencies[: N // 2], amplitude[: N // 2]
    )  # Afficher uniquement les fréquences positives
    plt.title(name_plot)
    plt.xlabel("Fréquence (Hz)")
    plt.ylabel("Amplitude")
    plt.xlim(0, max_freq)  # Limiter l'affichage aux fréquences jusqu'à max_freq Hz
    plt.grid()
    plt.show()
    return frequencies, amplitude


def calculer_et_visualiser_stft(
    filename: str,
    name_plot: str = "STFT du violon",
    hop_length: int = 512,
    n_fft: int = 2048,
    max_freq: int = 5000,
    y_axis: str = "hz",
):
    y, sr = librosa.load(filename, sr=None)
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    plt.figure(figsize=(14, 5))
    librosa.display.specshow(
        S_db, sr=sr, hop_length=hop_length, x_axis="time", y_axis=y_axis
    )
    plt.colorbar(format="%+2.0f dB")
    plt.title(name_plot)
    plt.xlabel("Temps (s)")
    plt.ylabel("Fréquence (Hz)")
    plt.ylim(20, max_freq)
    plt.tight_layout()
    plt.show()


def caracteriser_signal(filename):
    # Charger le signal audio
    y, sr = librosa.load(filename, sr=None)

    # Extraire les caractéristiques du signal audio
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)

    # Construire un vecteur de caractéristiques
    features = np.hstack(
        [
            np.mean(mfcc, axis=1),
            np.mean(chroma, axis=1),
            np.mean(spectral_centroid, axis=1),
            np.mean(spectral_rolloff, axis=1),
            np.mean(zero_crossing_rate, axis=1),
        ]
    )

    return features
