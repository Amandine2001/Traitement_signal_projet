from scipy.signal import butter, lfilter
import librosa
import numpy as np
import matplotlib.pyplot as plt


class bandeSonore:
    def __init__(self, name, filePath):
        self.name = name
        self.y, self.sr = self.load_sound(filePath)
        self.centroid = self.spectral_centroid()
        self.fournier = self.calcul_stft()
        self.mfcc = self.calcul_mfcc()
        self.chroma = self.calcul_chroma()
        self.dominant_note = self.detecter_note_dominante()
        self.tempo = self.detecter_tempo()
        self.zcr = self.zcr()
        self.features = self.calcul_features_vecteur()
        self.harmonic = None
        self.percussive = None

    ## CHARGEMENT DU SON
    def load_sound(self, filePath):
        y, sr = librosa.load(filePath, sr=None)
        return y, sr

    ## AFFICHAGE DES CARACTERISTIQUES DU SIGNAL
    def describe(self):
        print(f"Nom : {self.name}")
        print(f"DurÃ©e du signal : {len(self.y) / self.sr:.2f} secondes")
        print(f"FrÃ©quence d'Ã©chantillonnage : {self.sr} Hz")
        print(f"Nombre d'Ã©chantillons : {len(self.y)}")
        print(f"Shape du signal : {self.y.shape}")
        print(f"Spectral Centroid : {self.centroid:.2f} Hz")
        print(f"MFCC moyenne: {self.mfcc.mean():.2f}")
        print(f"Note dominante : {self.dominant_note}")
        print(f"Chroma moyenne : {self.chroma.mean():.2f}")
        print(f"Tempo : {self.tempo:.2f} BPM")
        print("\n")
        print("-" * 40)
        print("\n")

    ## ECHANTILLONNAGE
    def echantillonner(self, startTime, stopTime):
        start_sample = int(startTime * self.sr)
        stop_sample = int(stopTime * self.sr)
        return self.y[start_sample:stop_sample]

    ## CALCUL DU SPECTRAL CENTROID
    def spectral_centroid(self):
        return librosa.feature.spectral_centroid(y=self.y, sr=self.sr).mean()

    ## CALCUL TRANSFORMEE DE FOURIER
    def calcul_stft(self):
        D = librosa.stft(self.y)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        return S_db

    ## CALCUL DES MFCC
    def calcul_mfcc(self):
        mfcc = librosa.feature.mfcc(y=self.y, sr=self.sr)
        return mfcc

    ## CALCUL DES CHROMA
    def calcul_chroma(self):
        chroma = librosa.feature.chroma_stft(y=self.y, sr=self.sr)
        return chroma

    ## DETECTION DU TEMPO
    def detecter_tempo(self):
        tempo, _ = librosa.beat.beat_track(y=self.y, sr=self.sr)
        return tempo

    ## DETECTION DE LA NOTE DOMINANTE
    def detecter_note_dominante(self):
        chroma_mean = self.chroma.mean(axis=1)
        dominant_note = librosa.midi_to_note(np.argmax(chroma_mean))
        return dominant_note

    ## SEPARATION HARMONIQUE ET PERCUSSIVE
    def separation_hpss(self):
        self.harmonic, self.percussive = librosa.effects.hpss(self.y)
        return self.harmonic, self.percussive

    ## ZERO CROSSING RATE
    def zcr(self):
        zcr = librosa.feature.zero_crossing_rate(y=self.y)
        return zcr

    ## VECTEUR DE CARACTERISTIQUES
    def calcul_features_vecteur(self):
        features = np.hstack(
            [
                np.mean(self.mfcc, axis=1),
                np.mean(self.chroma, axis=1),
                np.mean(self.centroid),
                np.mean(self.zcr),
            ]
        )
        return features

    # ------------------------------------------------------------------------------------------------

    ## AFFICHAGE DE LA FORME D'ONDE
    def afficher_forme_onde(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        librosa.display.waveshow(self.y, sr=self.sr, ax=ax)
        ax.set_title(f"Forme d'onde - {self.name}")

    ## AFFICHAGE DU SPECTROGRAMME
    def afficher_spectrogramme(
        self, startTime=None, stopTime=None, ax=None, valeurs=None, title_suffix=""
    ):
        if ax is None:
            fig, ax = plt.subplots()
        ax.set_title(f"{self.name} {title_suffix}")

        if startTime is not None and stopTime is not None:
            echantillon = self.echantillonner(startTime, stopTime)
        elif valeurs is not None:
            echantillon = valeurs
        else:
            echantillon = self.y

        D = librosa.stft(echantillon)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        librosa.display.specshow(S_db, sr=self.sr, x_axis="time", y_axis="log", ax=ax)

    ## AFFICHAGE DES MFCC
    def afficher_mfcc(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        librosa.display.specshow(self.mfcc, x_axis="time", ax=ax)
        ax.set_title(f"MFCC - {self.name}")

    ## AFFICHAGE DES CHROMA
    def afficher_chroma(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        librosa.display.specshow(self.chroma, x_axis="time", y_axis="chroma", ax=ax)
        ax.set_title(f"Chroma - {self.name} - Note dominante : {self.dominant_note}")

    ## AFFICHAGE DE LA SEPARATION HARMONIQUE ET PERCUSSIVE
    def afficher_hpss(self, ax=None):
        if self.harmonic is None or self.percussive is None:
            self.separation_hpss()
        if ax is None:
            fig, ax = plt.subplots(2, 1, figsize=(10, 6))
        else:
            fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

        self.afficher_spectrogramme(
            valeurs=self.harmonic, ax=ax[0], title_suffix=" - Composante harmonique"
        )
        self.afficher_spectrogramme(
            valeurs=self.percussive, ax=ax[1], title_suffix=" - Composante percussive"
        )

    ## TRAITEMENTS DU SON

    # FILTRE PASSE BAS
    def passe_bas(self, cutoff, order=5):
        normal_cutoff = cutoff / (0.5 * self.sr)
        b, a = butter(order, normal_cutoff, btype="low", analog=False)
        return lfilter(b, a, self.y)

    def passe_haut(self, cutoff, order=5):
        normal_cutoff = cutoff / (0.5 * self.sr)
        b, a = butter(order, normal_cutoff, btype="high", analog=False)
        return lfilter(b, a, self.y)
