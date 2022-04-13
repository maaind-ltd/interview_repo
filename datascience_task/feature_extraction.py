'''
Problem statement
Given a function extract_features (provided, below) that does feature extraction (of MFCCs and Mel coefficients) of wav audio files, implement a short pipeline that:
takes a voice sample from the speech sample database  http://emodb.bilderbar.info/index-1280.html
group those samples based on gender
and extract MFCC and mel from them.

Can you find any meaningful differences between the groups? You can interpret ‘meaningful’ in different ways and there isn’t a single best or correct answer here. Use Data Science best practices wherever relevant.

Feel free to move this into a Jupyter notebook if you prefer.
'''

import soundfile
import librosa
import numpy as np
from scipy.io import wavfile


def extract_feature(file_name, **kwargs):
    """
    Extract feature from audio file `file_name`
        Features supported:
            - MFCC (mfcc)
            - Chroma (chroma)
            - MEL Spectrogram Frequency (mel)
            - Contrast (contrast)
            - Tonnetz (tonnetz)
        e.g:
        `features = extract_feature(path, mel=True, mfcc=True)`
    """
    mfcc = kwargs.get("mfcc")
    chroma = kwargs.get("chroma")
    mel = kwargs.get("mel")
    contrast = kwargs.get("contrast")
    tonnetz = kwargs.get("tonnetz")

    Fs, X = wavfile.read(file_name)
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        if chroma or contrast:
            stft = np.abs(librosa.stft(X))
        result = {}
        if mfcc:
            mfccs = np.mean(
                librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,
                axis=0)
            result['mfcc'] = mfccs
        if chroma:
            chroma = np.mean(
                librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,
                axis=0)
            result['chroma'] = chroma
        if mel:
            mel = np.mean(
                librosa.feature.melspectrogram(X, sr=sample_rate).T,
                axis=0)
            result['mel'] = mel
        if contrast:
            contrast = np.mean(
                librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,
                axis=0)
            result['contrast'] = contrast
        if tonnetz:
            tonnetz = np.mean(
                librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,
                axis=0)
            result['tonnetz'] = tonnetz

    return result
