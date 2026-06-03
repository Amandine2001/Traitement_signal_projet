"""Outils d'analyse et de visualisation de signaux audio.

La classe `bandeSonore` regroupe le chargement d'un fichier sonore,
le calcul de caractéristiques audio courantes et quelques méthodes
d'affichage utiles pour documenter un projet de traitement du signal.
"""

import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter

  

class bandeSonore:
    """Représente un signal audio et ses caractéristiques extraites."""

    def __init__(self, name, filePath):
        """Charge le son et calcule immédiatement les descripteurs principaux."""
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
        self.harmonic=None
        self.percussive=None

    # Chargement et caractéristiques de base
    def load_sound(self, filePath):
        """Charge un fichier audio via Librosa et renvoie le signal et sa fréquence d'échantillonnage."""
        y, sr = librosa.load(filePath, sr=None)
        return y, sr

    def describe(self):
        """Affiche un résumé lisible des caractéristiques extraites du signal."""
        print(f"Nom : {self.name}")
        print(f"DurÃ©e du signal : {len(self.y)/self.sr:.2f} secondes")
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


    def echantillonner(self, startTime, stopTime):
        """Retourne une portion du signal entre deux instants, en secondes."""
        start_sample = int(startTime * self.sr)
        stop_sample = int(stopTime * self.sr)
        return self.y[start_sample:stop_sample]

    def spectral_centroid(self):
        """Calcule le centroïde spectral moyen du signal."""
        return librosa.feature.spectral_centroid(y=self.y, sr=self.sr).mean()

    def calcul_stft(self):
        """Calcule le spectrogramme STFT en décibels."""
        D = librosa.stft(self.y)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        return S_db

    def calcul_mfcc(self):
        """Calcule les coefficients MFCC du signal."""
        mfcc = librosa.feature.mfcc(y=self.y, sr=self.sr)
        return mfcc

    def calcul_chroma(self):
        """Calcule la représentation chroma du signal."""
        chroma = librosa.feature.chroma_stft(y=self.y, sr=self.sr)
        return chroma

    def detecter_tempo(self):
        """Estime le tempo principal en BPM."""
        tempo, _ = librosa.beat.beat_track(y=self.y, sr=self.sr)
        return tempo
    
    def detecter_note_dominante(self):
        """Déduit la note la plus présente à partir de la moyenne chroma."""
        chroma_mean = self.chroma.mean(axis=1)
        dominant_note = librosa.midi_to_note(np.argmax(chroma_mean))
        return dominant_note

    def separation_hpss(self):
        """Sépare le signal en composantes harmonique et percussive."""
        self.harmonic, self.percussive = librosa.effects.hpss(self.y)
        return self.harmonic, self.percussive

    def zcr(self):
        """Calcule le taux moyen de passages par zéro."""
        zcr = librosa.feature.zero_crossing_rate(y=self.y)
        return zcr
    
    def calcul_features_vecteur(self):
        """Concatène les principaux descripteurs audio dans un vecteur unique."""
        features = np.hstack([ 
            np.mean(self.mfcc, axis=1), 
            np.mean(self.chroma, axis=1), 
            np.mean(self.centroid), 
            self.tempo,
            self.dominant_note,
            np.mean(self.zcr) 
        ]) 
        return features
#------------------------------------------------------------------------------------------------

    def afficher_forme_onde(self, ax=None):
        """Affiche la forme d'onde du signal."""
        if ax is None:
            fig, ax = plt.subplots()
        librosa.display.waveshow(self.y, sr=self.sr, ax=ax)
        ax.set_title(f'Forme d\'onde - {self.name}')
        
    def afficher_spectrogramme(self, startTime=None, stopTime=None, ax=None, valeurs=None, title_suffix=''):
        """Affiche un spectrogramme à partir du signal complet ou d'un extrait."""
        if ax is None:
            fig, ax = plt.subplots()
        ax.set_title(f'{self.name} {title_suffix}')

        if startTime is not None and stopTime is not None:
            echantillon = self.echantillonner(startTime, stopTime)
        elif valeurs is not None:
            echantillon = valeurs
        else:
            echantillon = self.y
        
        D = librosa.stft(echantillon)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        librosa.display.specshow(S_db, sr=self.sr, x_axis='time', y_axis='log', ax=ax)
    

    def afficher_mfcc(self, ax=None):        
        """Affiche la matrice MFCC."""
        if ax is None:
            fig, ax = plt.subplots()
        librosa.display.specshow(self.mfcc, x_axis='time', ax=ax)
        ax.set_title(f'MFCC - {self.name}')
       
    
    def afficher_chroma(self, ax=None):
        """Affiche la représentation chroma et la note dominante."""
        if ax is None:
            fig, ax = plt.subplots()
        librosa.display.specshow(self.chroma, x_axis='time', y_axis='chroma', ax=ax)
        ax.set_title(f'Chroma - {self.name} - Note dominante : {self.dominant_note}')
    
    def afficher_hpss(self, ax=None):
        """Affiche les spectrogrammes des composantes harmonique et percussive."""
        if self.harmonic is None or self.percussive is None:
            self.separation_hpss()
        if ax is None:
            fig, ax = plt.subplots(2, 1, figsize=(10, 6))
        else:
            fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        
        self.afficher_spectrogramme(valeurs=self.harmonic, ax=ax[0], title_suffix=' - Composante harmonique')
        self.afficher_spectrogramme(valeurs=self.percussive, ax=ax[1], title_suffix=' - Composante percussive')


    def passe_bas(self, cutoff, order=5):
        """Applique un filtre passe-bas de Butterworth."""
        normal_cutoff = cutoff / (0.5*self.sr)
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return lfilter(b, a, self.y)
    
    def passe_haut(self, cutoff, order=5):
        """Applique un filtre passe-haut de Butterworth."""
        normal_cutoff = cutoff / (0.5*self.sr)
        b, a = butter(order, normal_cutoff, btype='high', analog=False)
        return lfilter(b, a, self.y)

        
